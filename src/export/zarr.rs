//! Export a `.tet` file to a Zarr v3 directory store.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::Serialize;
use serde_json::json;

use crate::catalog::{
    CHUNK_PAYLOAD_CODEC_V1, CatalogError, ChunkIndexEntryV1, DatasetRecordV1, TetFileSummaryV1,
    read_tet_summary_v1, usize_from_u64,
};
use crate::layout::{self, mmap_file_read};
use crate::utils::dtype::ElementDtype;

/// Export a mapped `.tet` to a Zarr v3 directory store.
///
/// # Errors
///
/// Returns [`ExportError`] when the file cannot be read, `output` is non-empty, or a dataset dtype
/// is unsupported for Zarr v3.
pub fn export_tet_to_zarr(input: &Path, output: &Path) -> Result<ExportReport, ExportError> {
    export_tet_to_zarr_with_progress(input, output, None::<fn(ExportProgress)>)
}

/// Like [`export_tet_to_zarr`], invoking `progress` after each chunk file is written.
///
/// # Errors
///
/// Same as [`export_tet_to_zarr`].
pub fn export_tet_to_zarr_with_progress(
    input: &Path,
    output: &Path,
    mut progress: Option<impl FnMut(ExportProgress)>,
) -> Result<ExportReport, ExportError> {
    let started = Instant::now();
    ensure_empty_output_dir(output)?;
    let mmap = mmap_file_read(input).map_err(|e| ExportError::Zarr(e.to_string()))?;
    let summary = read_tet_summary_v1(&mmap).map_err(ExportError::Catalog)?;
    if summary.datasets.is_empty() {
        return Err(ExportError::NoDatasets {
            path: input.display().to_string(),
        });
    }

    write_root_group(output)?;
    let chunks_total = summary.chunks.len() as u64;

    let mut chunks_done = 0u64;
    let mut dataset_names = Vec::with_capacity(summary.datasets.len());
    for (ds_id, ds) in summary.datasets.iter().enumerate() {
        dataset_names.push(ds.name.clone());
        let wire_id = u64::try_from(ds_id).map_err(|_| {
            ExportError::Catalog(CatalogError::TooLargeForPlatform {
                field: "dataset_id",
                value: ds_id as u64,
            })
        })?;
        let chunks: Vec<&ChunkIndexEntryV1> = summary
            .chunks
            .iter()
            .filter(|c| c.dataset_id == wire_id)
            .collect();
        export_dataset(&mmap, output, ds, &chunks, &summary)?;
        for _ in &chunks {
            chunks_done += 1;
            if let Some(ref mut cb) = progress {
                cb(ExportProgress {
                    chunks_done,
                    chunks_total,
                    dataset: ds.name.clone(),
                });
            }
        }
    }

    Ok(ExportReport {
        input: input.display().to_string(),
        output: output.display().to_string(),
        dataset_count: summary.datasets.len(),
        dataset_names,
        chunks_written: chunks_done,
        elapsed_secs: started.elapsed().as_secs_f64(),
    })
}

/// Progress event emitted while writing chunk files.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct ExportProgress {
    pub chunks_done: u64,
    pub chunks_total: u64,
    pub dataset: String,
}

/// Summary returned after a successful export.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct ExportReport {
    pub input: String,
    pub output: String,
    pub dataset_count: usize,
    pub dataset_names: Vec<String>,
    pub chunks_written: u64,
    pub elapsed_secs: f64,
}

/// Export pipeline failures.
#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    #[error(transparent)]
    Layout(#[from] layout::LayoutError),
    #[error(transparent)]
    Catalog(#[from] CatalogError),
    #[error("no datasets in {path}")]
    NoDatasets { path: String },
    #[error("export output `{path}` exists and is not empty")]
    OutputNotEmpty { path: String },
    #[error("Zarr export failed: {0}")]
    Zarr(String),
    #[error("unsupported dataset `{name}` dtype for Zarr export: {detail}")]
    UnsupportedDtype { name: String, detail: String },
}

fn ensure_empty_output_dir(path: &Path) -> Result<(), ExportError> {
    if path.exists() {
        let mut entries = fs::read_dir(path).map_err(|e| ExportError::Zarr(e.to_string()))?;
        if entries.next().is_some() {
            return Err(ExportError::OutputNotEmpty {
                path: path.display().to_string(),
            });
        }
    } else {
        fs::create_dir_all(path).map_err(|e| ExportError::Zarr(e.to_string()))?;
    }
    Ok(())
}

fn write_root_group(store: &Path) -> Result<(), ExportError> {
    let meta = json!({
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "tetration_export": env!("CARGO_PKG_VERSION"),
        }
    });
    write_json_file(&store.join("zarr.json"), &meta)
}

fn export_dataset(
    mmap: &[u8],
    store: &Path,
    ds: &DatasetRecordV1,
    chunks: &[&ChunkIndexEntryV1],
    summary: &TetFileSummaryV1,
) -> Result<(), ExportError> {
    let elem =
        ElementDtype::try_from_wire_tag(ds.dtype).ok_or_else(|| ExportError::UnsupportedDtype {
            name: ds.name.clone(),
            detail: format!("unsupported wire dtype tag {:#x}", ds.dtype),
        })?;
    let ndim = ds.shape.len();
    ensure_group_chain(store, &ds.name)?;
    let array_dir = array_dir_for_name(store, &ds.name);
    fs::create_dir_all(&array_dir).map_err(|e| ExportError::Zarr(e.to_string()))?;

    let zstd = chunks
        .first()
        .is_some_and(|c| CHUNK_PAYLOAD_CODEC_V1.is_zstd(c.codec));
    if chunks
        .iter()
        .any(|c| CHUNK_PAYLOAD_CODEC_V1.is_zstd(c.codec) != zstd)
    {
        return Err(ExportError::Zarr(format!(
            "dataset `{}` mixes raw and zstd chunk codecs; re-export with one codec",
            ds.name
        )));
    }

    let attrs = dataset_export_attrs(summary, &ds.name, elem);
    let meta = array_zarr_json(ds, elem, zstd, &attrs);
    write_json_file(&array_dir.join("zarr.json"), &meta)?;

    for entry in chunks {
        write_chunk_file(mmap, &array_dir, entry, ndim)?;
    }
    Ok(())
}

fn dataset_export_attrs(
    summary: &TetFileSummaryV1,
    name: &str,
    elem: ElementDtype,
) -> BTreeMap<String, serde_json::Value> {
    let mut attrs = BTreeMap::new();
    attrs.insert(
        "tetration_dtype".into(),
        json!(element_dtype_wire_label(elem)),
    );
    if let Some(ds_meta) = summary.metadata.datasets.get(name) {
        for (k, v) in &ds_meta.attrs {
            attrs.insert(k.clone(), json!(v));
        }
    }
    attrs
}

fn element_dtype_wire_label(elem: ElementDtype) -> &'static str {
    match elem {
        ElementDtype::F32 => "f32",
        ElementDtype::F64 => "f64",
        ElementDtype::I32 => "i32",
        ElementDtype::I64 => "i64",
        ElementDtype::U8 => "u8",
        ElementDtype::U16 => "u16",
        ElementDtype::I16 => "i16",
        ElementDtype::U32 => "u32",
        ElementDtype::F16 => "f16",
        ElementDtype::U64 => "u64",
    }
}

fn element_dtype_zarr_name(elem: ElementDtype) -> &'static str {
    match elem {
        ElementDtype::F32 => "float32",
        ElementDtype::F64 => "float64",
        ElementDtype::I32 => "int32",
        ElementDtype::I64 => "int64",
        ElementDtype::U8 => "uint8",
        ElementDtype::U16 => "uint16",
        ElementDtype::I16 => "int16",
        ElementDtype::U32 => "uint32",
        ElementDtype::F16 => "float16",
        ElementDtype::U64 => "uint64",
    }
}

fn array_zarr_json(
    ds: &DatasetRecordV1,
    elem: ElementDtype,
    zstd: bool,
    attributes: &BTreeMap<String, serde_json::Value>,
) -> serde_json::Value {
    let data_type = element_dtype_zarr_name(elem);
    let fill_value: serde_json::Value = if elem.is_integer() {
        json!(0)
    } else {
        json!(0.0)
    };
    let mut codecs = vec![json!({
        "name": "bytes",
        "configuration": { "endian": "little" }
    })];
    if zstd {
        codecs.push(json!({
            "name": "zstd",
            "configuration": { "level": 0, "checksum": false }
        }));
    }
    json!({
        "zarr_format": 3,
        "node_type": "array",
        "shape": ds.shape,
        "data_type": data_type,
        "chunk_grid": {
            "name": "regular",
            "configuration": { "chunk_shape": ds.chunk_shape }
        },
        "chunk_key_encoding": {
            "name": "default",
            "configuration": { "separator": "/" }
        },
        "fill_value": fill_value,
        "codecs": codecs,
        "attributes": attributes,
        "storage_transformers": []
    })
}

fn write_chunk_file(
    mmap: &[u8],
    array_dir: &Path,
    entry: &ChunkIndexEntryV1,
    ndim: usize,
) -> Result<(), ExportError> {
    let start = usize_from_u64("payload_offset", entry.payload_offset)?;
    let end = start
        .checked_add(usize_from_u64("stored_byte_len", entry.stored_byte_len)?)
        .ok_or_else(|| ExportError::Zarr("chunk payload span overflow".into()))?;
    if end > mmap.len() {
        return Err(ExportError::Catalog(CatalogError::PayloadOutOfBounds {
            file_len: mmap.len() as u64,
            start: entry.payload_offset,
            end: end as u64,
        }));
    }
    let payload = &mmap[start..end];
    if !CHUNK_PAYLOAD_CODEC_V1.is_supported(entry.codec) {
        return Err(ExportError::Catalog(CatalogError::UnsupportedCodec {
            codec: entry.codec,
        }));
    }

    let mut path = array_dir.join("c");
    for d in 0..ndim {
        path.push(entry.chunk_index[d].to_string());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| ExportError::Zarr(e.to_string()))?;
    }
    fs::write(&path, payload).map_err(|e| ExportError::Zarr(e.to_string()))?;
    Ok(())
}

fn ensure_group_chain(store: &Path, catalog_name: &str) -> Result<(), ExportError> {
    let parts: Vec<&str> = catalog_name.split('/').collect();
    if parts.len() <= 1 {
        return Ok(());
    }
    for i in 0..parts.len() - 1 {
        let rel = parts[..=i].join("/");
        let dir = array_dir_for_name(store, &rel);
        if dir.join("zarr.json").is_file() {
            continue;
        }
        fs::create_dir_all(&dir).map_err(|e| ExportError::Zarr(e.to_string()))?;
        let meta = json!({
            "zarr_format": 3,
            "node_type": "group",
        });
        write_json_file(&dir.join("zarr.json"), &meta)?;
    }
    Ok(())
}

fn array_dir_for_name(store: &Path, catalog_name: &str) -> PathBuf {
    if catalog_name.is_empty() {
        store.to_path_buf()
    } else {
        store.join(catalog_name)
    }
}

fn write_json_file(path: &Path, value: &serde_json::Value) -> Result<(), ExportError> {
    let raw = serde_json::to_string_pretty(value).map_err(|e| ExportError::Zarr(e.to_string()))?;
    fs::write(path, raw).map_err(|e| ExportError::Zarr(e.to_string()))
}
