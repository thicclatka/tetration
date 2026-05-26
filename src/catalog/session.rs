//! Embedder-oriented create / open helpers (Phase 7).
//!
//! [`TetWriterSession`] buffers datasets and optional history rows, then writes one `.tet` on
//! [`TetWriterSession::commit`] (in-memory tensors) or [`TetWriterSession::commit_with_fill`]
//! (streaming tiles). [`TetFile`] keeps the backing file open for mmap query execution.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;

use crate::layout::{self, SuperblockV1, mmap_file_read};
use crate::utils::dtype::ElementDtype;

use super::dataset::RawArrayWrite;
use super::execution::FileExecutionSettingsV1;
use super::history::{self, FooterBlobV1, HistoryEventV1, write_footer_blob};
use super::metadata::{DatasetMetadataV1, FileMetadataV1, TetMetadataV1};
use super::stream_write::{ArrayWriteMeta, StreamTileJob, write_multi_raw_array_streaming};
use super::tile;
use super::{
    CHUNK_PAYLOAD_CODEC_V1, CatalogError, DATASET_DTYPE_TAG_V1, DatasetRecordV1, TetFileSummaryV1,
    read_tet_summary_v1, validate_array_write_meta, write_multi_raw_array_file,
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

/// Streaming dataset spec: geometry + footer metadata only (tile bytes supplied at commit).
#[derive(Debug, Clone)]
pub struct TetDatasetStreamSpec {
    pub name: String,
    pub dtype: u32,
    pub shape: Vec<u64>,
    pub chunk_shape: Vec<u64>,
    pub attrs: BTreeMap<String, String>,
    pub dim_names: Option<Vec<String>>,
}

impl TetDatasetStreamSpec {
    /// Row-major `f32` grid with raw chunk codec (**0**); validate shape/chunk grid only.
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError`] when `shape` / `chunk_shape` are inconsistent.
    pub fn f32_row_major(
        name: impl Into<String>,
        shape: &[u64],
        chunk_shape: &[u64],
    ) -> Result<Self, CatalogError> {
        let name = name.into();
        let meta = ArrayWriteMeta {
            name: &name,
            dtype: DATASET_DTYPE_TAG_V1.f32,
            shape,
            chunk_shape,
            chunk_codec: CHUNK_PAYLOAD_CODEC_V1.raw,
            file_execution: None,
        };
        validate_array_write_meta(&meta)?;
        Ok(Self {
            name,
            dtype: DATASET_DTYPE_TAG_V1.f32,
            shape: shape.to_vec(),
            chunk_shape: chunk_shape.to_vec(),
            attrs: BTreeMap::new(),
            dim_names: None,
        })
    }
}

/// Draft file-level metadata; mapped into [`FileMetadataV1`] on commit.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FileMetadataDraft {
    pub tool: Option<String>,
    pub library_version: Option<String>,
}

#[derive(Debug, Clone)]
enum SessionDataset {
    InMemory(TetDatasetWrite),
    Streaming(TetDatasetStreamSpec),
}

impl SessionDataset {
    fn name(&self) -> &str {
        match self {
            Self::InMemory(d) => &d.name,
            Self::Streaming(d) => &d.name,
        }
    }

    fn attrs(&self) -> &BTreeMap<String, String> {
        match self {
            Self::InMemory(d) => &d.attrs,
            Self::Streaming(d) => &d.attrs,
        }
    }

    fn dim_names(&self) -> &Option<Vec<String>> {
        match self {
            Self::InMemory(d) => &d.dim_names,
            Self::Streaming(d) => &d.dim_names,
        }
    }

    fn is_streaming(&self) -> bool {
        matches!(self, Self::Streaming(_))
    }
}

/// Buffered writer: queue datasets and history, flush on commit.
#[derive(Debug, Clone)]
pub struct TetWriterSession {
    path: PathBuf,
    datasets: Vec<SessionDataset>,
    history: Vec<HistoryEventV1>,
    file_execution: Option<FileExecutionSettingsV1>,
    pub metadata: FileMetadataDraft,
}

impl TetWriterSession {
    /// New session targeting `path` (file is created on commit, truncating any prior file).
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

    /// Queue an in-memory dataset (validated immediately).
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
        self.datasets.push(SessionDataset::InMemory(dataset));
        Ok(())
    }

    /// Queue a streaming dataset (geometry validated; tile bytes supplied in [`Self::commit_with_fill`]).
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError`] when the write spec is invalid.
    pub fn push_dataset_streaming(
        &mut self,
        spec: TetDatasetStreamSpec,
    ) -> Result<(), CatalogError> {
        let meta = ArrayWriteMeta {
            name: &spec.name,
            dtype: spec.dtype,
            shape: &spec.shape,
            chunk_shape: &spec.chunk_shape,
            chunk_codec: CHUNK_PAYLOAD_CODEC_V1.raw,
            file_execution: None,
        };
        validate_array_write_meta(&meta)?;
        self.datasets.push(SessionDataset::Streaming(spec));
        Ok(())
    }

    /// Append a history row flushed on commit (`op`, `source`, Unix seconds as decimal string).
    pub fn push_history_event(&mut self, op: impl Into<String>, source: impl Into<String>) {
        self.history
            .push((op.into(), source.into(), history::unix_timestamp_now()));
    }

    fn has_streaming(&self) -> bool {
        self.datasets.iter().any(SessionDataset::is_streaming)
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
            if ds.attrs().is_empty() && ds.dim_names().is_none() {
                continue;
            }
            let entry = meta.dataset_mut(ds.name());
            entry.attrs = ds.attrs().clone();
            entry.dim_names.clone_from(ds.dim_names());
        }
        if meta.file.is_none() && meta.datasets.is_empty() {
            return Ok(None);
        }
        meta.validate()?;
        Ok(Some(meta))
    }

    fn flush_footer(&self, metadata: Option<TetMetadataV1>) -> Result<(), CatalogError> {
        if !self.history.is_empty() || metadata.is_some() {
            write_footer_blob(
                &self.path,
                &FooterBlobV1 {
                    history: self.history.clone(),
                    metadata,
                },
            )?;
        }
        Ok(())
    }

    /// Write the `.tet` when every queued dataset is in-memory.
    ///
    /// Streaming datasets require [`Self::commit_with_fill`].
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError`] when no datasets were queued, a streaming dataset is present,
    /// layout validation fails, or I/O fails.
    pub fn commit(self) -> Result<PathBuf, CatalogError> {
        if self.datasets.is_empty() {
            return Err(CatalogError::InvalidWriteSpec(
                "TetWriterSession: at least one dataset is required",
            ));
        }
        if self.has_streaming() {
            return Err(CatalogError::InvalidWriteSpec(
                "TetWriterSession: streaming datasets require commit_with_fill",
            ));
        }
        let specs: Vec<RawArrayWrite<'_>> = self
            .datasets
            .iter()
            .map(|d| {
                let d = match d {
                    SessionDataset::InMemory(w) => w,
                    SessionDataset::Streaming(_) => unreachable!(),
                };
                RawArrayWrite {
                    name: &d.name,
                    dtype: d.dtype,
                    shape: &d.shape,
                    chunk_shape: &d.chunk_shape,
                    chunk_codec: d.chunk_codec,
                    data: &d.data,
                    file_execution: self.file_execution,
                }
            })
            .collect();
        write_multi_raw_array_file(&self.path, &specs)?;
        self.flush_footer(self.build_footer_metadata()?)?;
        Ok(self.path)
    }

    /// Write the `.tet`, filling streaming tiles via `fill` (raw codec only).
    ///
    /// In-memory datasets are sliced automatically; `fill` is invoked only for
    /// [`TetDatasetStreamSpec`] entries. When every dataset is in-memory, behaves like
    /// [`Self::commit`] and ignores `fill`.
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError`] when no datasets were queued, layout validation fails, `fill`
    /// fails, or I/O fails.
    pub fn commit_with_fill<F>(
        mut self,
        parallel_jobs: usize,
        fill: F,
    ) -> Result<PathBuf, CatalogError>
    where
        F: Fn(&StreamTileJob<'_>, &mut [u8]) -> Result<(), CatalogError> + Sync + Send,
    {
        if self.datasets.is_empty() {
            return Err(CatalogError::InvalidWriteSpec(
                "TetWriterSession: at least one dataset is required",
            ));
        }
        if !self.has_streaming() {
            return self.commit();
        }

        let footer_metadata = self.build_footer_metadata()?;

        struct Prepared {
            name: String,
            dtype: u32,
            shape: Vec<u64>,
            chunk_shape: Vec<u64>,
            data: Option<Vec<u8>>,
        }

        let file_execution = self.file_execution;
        let mut prepared: Vec<Prepared> = Vec::with_capacity(self.datasets.len());
        for ds in std::mem::take(&mut self.datasets) {
            match ds {
                SessionDataset::InMemory(w) => prepared.push(Prepared {
                    name: w.name,
                    dtype: w.dtype,
                    shape: w.shape,
                    chunk_shape: w.chunk_shape,
                    data: Some(w.data),
                }),
                SessionDataset::Streaming(s) => prepared.push(Prepared {
                    name: s.name,
                    dtype: s.dtype,
                    shape: s.shape,
                    chunk_shape: s.chunk_shape,
                    data: None,
                }),
            }
        }

        let metas: Vec<ArrayWriteMeta<'_>> = prepared
            .iter()
            .map(|d| ArrayWriteMeta {
                name: &d.name,
                dtype: d.dtype,
                shape: &d.shape,
                chunk_shape: &d.chunk_shape,
                chunk_codec: CHUNK_PAYLOAD_CODEC_V1.raw,
                file_execution,
            })
            .collect();

        write_multi_raw_array_streaming(
            &self.path,
            &metas,
            parallel_jobs,
            |job, buf| {
                let ds = &prepared[job.dataset_id];
                if let Some(data) = &ds.data {
                    let elem = ElementDtype::try_from_wire_tag(ds.dtype).ok_or(
                        CatalogError::InvalidWriteSpec("unsupported dataset dtype tag"),
                    )?;
                    let tile = tile::extract_tile_row_major(
                        data,
                        &ds.shape,
                        &ds.chunk_shape,
                        &job.chunk_coord[..job.ndim],
                        job.ndim,
                        elem.elem_size(),
                    )?;
                    if tile.len() != buf.len() {
                        return Err(CatalogError::InvalidWriteSpec(
                            "in-memory tile length mismatch for chunk",
                        ));
                    }
                    buf.copy_from_slice(&tile);
                    Ok(())
                } else {
                    fill(job, buf)
                }
            },
            None,
        )?;
        self.flush_footer(footer_metadata)?;
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
