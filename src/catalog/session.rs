//! Embedder-oriented create / open helpers (Phase 7).
//!
//! [`TetWriterSession`] buffers datasets and optional history rows, then writes one `.tet` on
//! [`TetWriterSession::commit`]. [`TetFile`] keeps the backing file open for mmap query execution.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;

use crate::layout::{self, SuperblockV1, mmap_file_read};

use super::dataset::RawArrayWrite;
use super::execution::FileExecutionSettingsV1;
use super::history::{self, FooterBlobV1, HistoryEventV1, write_footer_blob};
use super::metadata::{DatasetMetadataV1, FileMetadataV1, TetMetadataV1};
use super::{
    CHUNK_PAYLOAD_CODEC_V1, CatalogError, DATASET_DTYPE_TAG_V1, DatasetRecordV1, TetFileSummaryV1,
    read_tet_summary_v1, write_multi_raw_array_file,
};

/// In-memory dataset queued for [`TetWriterSession::commit`].
#[derive(Debug, Clone)]
pub struct TetDatasetWrite {
    pub name: String,
    pub dtype: u32,
    pub shape: Vec<u64>,
    pub chunk_shape: Vec<u64>,
    pub chunk_codec: u32,
    pub data: Vec<u8>,
    /// CF-style string attributes persisted in the footer `metadata` object.
    pub attrs: BTreeMap<String, String>,
    /// Optional dimension names (`ndim` strings); coordinate labels are not stored yet.
    pub dim_names: Option<Vec<String>>,
}

impl TetDatasetWrite {
    /// Row-major `f32` tensor with raw chunk codec (**0**).
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError`] when `shape` / `chunk_shape` / `data` are inconsistent.
    pub fn f32_row_major(
        name: impl Into<String>,
        shape: &[u64],
        chunk_shape: &[u64],
        data: Vec<u8>,
    ) -> Result<Self, CatalogError> {
        let name = name.into();
        let spec = RawArrayWrite {
            name: &name,
            dtype: DATASET_DTYPE_TAG_V1.f32,
            shape,
            chunk_shape,
            chunk_codec: CHUNK_PAYLOAD_CODEC_V1.raw,
            data: &data,
            file_execution: None,
        };
        super::dataset::validate_raw_array_write(&spec)?;
        Ok(Self {
            name,
            dtype: DATASET_DTYPE_TAG_V1.f32,
            shape: shape.to_vec(),
            chunk_shape: chunk_shape.to_vec(),
            chunk_codec: CHUNK_PAYLOAD_CODEC_V1.raw,
            data,
            attrs: BTreeMap::new(),
            dim_names: None,
        })
    }
}

/// Draft file-level metadata; mapped into [`FileMetadataV1`] on [`TetWriterSession::commit`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FileMetadataDraft {
    pub tool: Option<String>,
    pub library_version: Option<String>,
}

/// Buffered writer: queue datasets and history, flush on [`Self::commit`].
#[derive(Debug, Clone)]
pub struct TetWriterSession {
    path: PathBuf,
    datasets: Vec<TetDatasetWrite>,
    history: Vec<HistoryEventV1>,
    file_execution: Option<FileExecutionSettingsV1>,
    pub metadata: FileMetadataDraft,
}

impl TetWriterSession {
    /// New session targeting `path` (file is created on [`Self::commit`], truncating any prior file).
    #[must_use]
    pub fn create(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            datasets: Vec::new(),
            history: Vec::new(),
            file_execution: None,
            metadata: FileMetadataDraft::default(),
        }
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    #[must_use]
    pub fn dataset_count(&self) -> usize {
        self.datasets.len()
    }

    /// Chunk-index execution settings applied when the file has at least one dataset.
    #[must_use]
    pub fn file_execution(mut self, settings: FileExecutionSettingsV1) -> Self {
        self.file_execution = Some(settings);
        self
    }

    /// Queue a dataset (validated immediately).
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError`] when the write spec is invalid.
    pub fn push_dataset(&mut self, dataset: TetDatasetWrite) -> Result<(), CatalogError> {
        let name = dataset.name.as_str();
        let spec = RawArrayWrite {
            name,
            dtype: dataset.dtype,
            shape: &dataset.shape,
            chunk_shape: &dataset.chunk_shape,
            chunk_codec: dataset.chunk_codec,
            data: &dataset.data,
            file_execution: None,
        };
        super::dataset::validate_raw_array_write(&spec)?;
        self.datasets.push(dataset);
        Ok(())
    }

    /// Append a history row flushed on commit (`op`, `source`, Unix seconds as decimal string).
    pub fn push_history_event(&mut self, op: impl Into<String>, source: impl Into<String>) {
        self.history
            .push((op.into(), source.into(), history::unix_timestamp_now()));
    }

    fn build_footer_metadata(&self) -> Result<Option<TetMetadataV1>, CatalogError> {
        let mut meta = TetMetadataV1::default();
        let has_file = self.metadata.tool.is_some() || self.metadata.library_version.is_some();
        if has_file {
            meta.file = Some(FileMetadataV1 {
                tool: self.metadata.tool.clone(),
                library_version: self
                    .metadata
                    .library_version
                    .clone()
                    .or_else(|| Some(env!("CARGO_PKG_VERSION").to_owned())),
                created_at: Some(history::unix_timestamp_now()),
            });
        }
        for ds in &self.datasets {
            if ds.attrs.is_empty() && ds.dim_names.is_none() {
                continue;
            }
            let entry = meta.dataset_mut(&ds.name);
            entry.attrs = ds.attrs.clone();
            entry.dim_names.clone_from(&ds.dim_names)
        }
        if meta.file.is_none() && meta.datasets.is_empty() {
            return Ok(None);
        }
        meta.validate()?;
        Ok(Some(meta))
    }

    /// Write the `.tet` and optional `THST` footer (history + metadata); returns the output path.
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError`] when no datasets were queued, layout validation fails, or I/O fails.
    pub fn commit(self) -> Result<PathBuf, CatalogError> {
        if self.datasets.is_empty() {
            return Err(CatalogError::InvalidWriteSpec(
                "TetWriterSession: at least one dataset is required",
            ));
        }
        let specs: Vec<RawArrayWrite<'_>> = self
            .datasets
            .iter()
            .map(|d| RawArrayWrite {
                name: &d.name,
                dtype: d.dtype,
                shape: &d.shape,
                chunk_shape: &d.chunk_shape,
                chunk_codec: d.chunk_codec,
                data: &d.data,
                file_execution: self.file_execution,
            })
            .collect();
        write_multi_raw_array_file(&self.path, &specs)?;
        let metadata = self.build_footer_metadata()?;
        if !self.history.is_empty() || metadata.is_some() {
            write_footer_blob(
                &self.path,
                &FooterBlobV1 {
                    history: self.history,
                    metadata,
                },
            )?;
        }
        Ok(self.path)
    }
}

/// Open `.tet` for mmap reads and query execution.
pub struct TetFile {
    path: PathBuf,
    _file: File,
    mmap: Mmap,
}

impl TetFile {
    /// Memory-map an existing file read-only.
    ///
    /// # Errors
    ///
    /// Propagates I/O errors from open or mmap.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, std::io::Error> {
        let path = path.into();
        let file = File::open(&path)?;
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
        Ok(Self {
            path,
            _file: file,
            mmap,
        })
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    #[must_use]
    pub fn mmap(&self) -> &[u8] {
        &self.mmap
    }

    /// Parsed superblock + catalog (+ footer history/metadata when present).
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError`] when layout rules are violated.
    pub fn summary(&self) -> Result<TetFileSummaryV1, CatalogError> {
        read_tet_summary_v1(self.mmap())
    }

    /// # Errors
    ///
    /// Returns [`layout::LayoutError`] when bytes at offset 0 are not a valid v1 superblock.
    pub fn superblock(&self) -> Result<SuperblockV1, layout::LayoutError> {
        layout::read_superblock_v1(self.mmap())
    }

    /// Dataset catalog records.
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError`] when the catalog cannot be read.
    pub fn datasets(&self) -> Result<Vec<DatasetRecordV1>, CatalogError> {
        Ok(self.summary()?.datasets)
    }

    /// Metadata for a dataset by catalog name.
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError`] when the catalog cannot be read.
    pub fn dataset_metadata(&self, name: &str) -> Result<Option<DatasetMetadataV1>, CatalogError> {
        Ok(self.summary()?.metadata.datasets.get(name).cloned())
    }
}

impl TetFile {
    /// Convenience: mmap from `path` without keeping [`TetFile`] alive.
    ///
    /// Prefer [`Self::open`] when running multiple queries on the same file.
    ///
    /// # Errors
    ///
    /// Propagates I/O errors from [`mmap_file_read`].
    pub fn mmap_bytes(path: impl AsRef<Path>) -> Result<Vec<u8>, std::io::Error> {
        let mmap = mmap_file_read(path.as_ref())?;
        Ok(mmap.to_vec())
    }
}
