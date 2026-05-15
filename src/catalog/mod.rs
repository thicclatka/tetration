//! Dataset directory and chunk index (layout v1 extension).
//!
//! See `docs/layout_v1.md` for byte layout after the 32-byte superblock.

mod dataset;
mod index;
mod tile;

use std::fs::OpenOptions;
use std::io::{self, Write};
use std::path::Path;

use serde::Serialize;
use thiserror::Error;

use crate::layout::{self, LAYOUT_VERSION_V1, SuperblockV1};
use crate::wire;

pub use dataset::{DatasetRecordV1, RawArrayWrite};
pub use index::{CHUNK_INDEX_HEADER_V1, ChunkIndexEntryV1, ChunkIndexHeaderV1};

/// Chunk grid coordinates whose tiles intersect the half-open global index box (see
/// [`tile::chunk_coords_intersecting_global_box`]).
///
/// # Errors
///
/// Returns [`CatalogError::InvalidWriteSpec`] when slice lengths disagree, the global box is
/// empty or out of bounds, or chunk-grid arithmetic overflows. Propagates other catalog tile
/// errors from the underlying helper.
pub fn chunk_coords_intersecting_global_box(
    shape: &[u64],
    chunk_shape: &[u64],
    g0: &[u64],
    g1_exclusive: &[u64],
) -> Result<Vec<[u64; MAX_NDIM]>, CatalogError> {
    tile::chunk_coords_intersecting_global_box(shape, chunk_shape, g0, g1_exclusive)
}

/// Chunk grid coordinates touching a per-axis strided selection (see
/// [`tile::chunk_coords_intersecting_strided`]).
///
/// # Errors
///
/// Same failure modes as [`chunk_coords_intersecting_global_box`], plus invalid `step` values.
pub fn chunk_coords_intersecting_strided(
    shape: &[u64],
    chunk_shape: &[u64],
    g0: &[u64],
    g1_exclusive: &[u64],
    step: &[u64],
) -> Result<Vec<[u64; MAX_NDIM]>, CatalogError> {
    tile::chunk_coords_intersecting_strided(shape, chunk_shape, g0, g1_exclusive, step)
}

/// `dtype` tag for IEEE754 binary32 elements (`f32`), row-major within a chunk.
pub const DTYPE_F32: u32 = 1;

/// Maximum tensor rank supported by the v1 catalog on disk.
pub const MAX_NDIM: usize = 8;

/// High-level view of a mapped `.tet` file (superblock + catalog).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct TetFileSummaryV1 {
    pub superblock: SuperblockV1,
    pub datasets: Vec<DatasetRecordV1>,
    pub chunks: Vec<ChunkIndexEntryV1>,
}

#[derive(Debug, Error)]
pub enum CatalogError {
    #[error(transparent)]
    Layout(#[from] layout::LayoutError),
    #[error("file too short for catalog: need at least {need} bytes, got {got}")]
    TooShort { need: usize, got: usize },
    #[error("dataset_count is {count} but dataset directory is missing (file length {len})")]
    MissingDatasetDirectory { count: u32, len: usize },
    #[error("invalid UTF-8 in dataset name")]
    BadDatasetName,
    #[error("ndim {ndim} out of range (max {MAX_NDIM})")]
    BadNdim { ndim: usize },
    #[error("dataset blob length mismatch: declared {declared}, parsed {parsed}")]
    DatasetBlobMismatch { declared: u64, parsed: u64 },
    #[error("chunk index offset mismatch: superblock says {sb}, expected {expected}")]
    ChunkIndexOffsetMismatch { sb: u64, expected: u64 },
    #[error("bad chunk index magic: expected TIDX, got {0:?}")]
    BadIndexMagic([u8; 4]),
    #[error("unsupported chunk index version: {0}")]
    UnsupportedIndexVersion(u32),
    #[error("chunk index entry count {count} does not fit in region length {region}")]
    BadIndexLength { count: u64, region: u64 },
    #[error("chunk payload [{start}, {end}) out of bounds for file length {file_len}")]
    PayloadOutOfBounds { file_len: u64, start: u64, end: u64 },
    #[error("unsupported codec {codec} (only 0 = raw is supported in this build)")]
    UnsupportedCodec { codec: u32 },
    #[error("raw/stored length mismatch for codec=0: raw={raw}, stored={stored}")]
    RawStoredMismatch { raw: u64, stored: u64 },
    #[error("invalid one-chunk write spec: {0}")]
    InvalidWriteSpec(&'static str),
    #[error("catalog numeric value too large for this platform ({field}={value})")]
    TooLargeForPlatform { field: &'static str, value: u64 },
    #[error(transparent)]
    Io(#[from] io::Error),
}

/// Arguments for writing a single-dataset, single-chunk raw payload file (no compression).
#[derive(Debug, Clone)]
pub struct OneChunkRawWrite<'a> {
    pub name: &'a str,
    pub dtype: u32,
    pub shape: &'a [u64],
    pub chunk_shape: &'a [u64],
    pub payload: &'a [u8],
}

/// Read superblock, dataset directory, and chunk index from a mapped file.
///
/// # Errors
///
/// Returns [`CatalogError`] when layout rules are violated or regions are inconsistent.
pub fn read_tet_summary_v1(data: &[u8]) -> Result<TetFileSummaryV1, CatalogError> {
    let sb = layout::read_superblock_v1(data)?;
    if sb.dataset_count == 0 {
        return Ok(TetFileSummaryV1 {
            superblock: sb,
            datasets: Vec::new(),
            chunks: Vec::new(),
        });
    }

    if data.len() < 40 {
        return Err(CatalogError::MissingDatasetDirectory {
            count: sb.dataset_count,
            len: data.len(),
        });
    }

    let dataset_blob_len_u64 = wire::u64_le_at(data, 32);
    let dataset_blob_len =
        usize::try_from(dataset_blob_len_u64).map_err(|_| CatalogError::TooLargeForPlatform {
            field: "dataset_blob_len",
            value: dataset_blob_len_u64,
        })?;
    let blob_start = 40usize;
    let blob_end = blob_start
        .checked_add(dataset_blob_len)
        .ok_or(CatalogError::TooShort {
            need: usize::MAX,
            got: data.len(),
        })?;
    if data.len() < blob_end {
        return Err(CatalogError::TooShort {
            need: blob_end,
            got: data.len(),
        });
    }

    let mut datasets = Vec::with_capacity(sb.dataset_count as usize);
    let mut cursor = blob_start;
    for _ in 0..sb.dataset_count {
        let (rec, next) = dataset::parse_one_dataset_record(data, cursor, blob_end)?;
        datasets.push(rec);
        cursor = next;
    }
    if cursor != blob_end {
        return Err(CatalogError::DatasetBlobMismatch {
            declared: dataset_blob_len as u64,
            parsed: (cursor - blob_start) as u64,
        });
    }

    let blob_end_u64 = u64::try_from(blob_end).map_err(|_| CatalogError::TooLargeForPlatform {
        field: "dataset_blob_end",
        value: u64::MAX,
    })?;
    let expected_index_off = wire::align8_u64(blob_end_u64);
    if sb.chunk_index_offset != expected_index_off {
        return Err(CatalogError::ChunkIndexOffsetMismatch {
            sb: sb.chunk_index_offset,
            expected: expected_index_off,
        });
    }

    let idx_start = usize_from_u64("chunk_index_offset", sb.chunk_index_offset)?;
    let idx_len = usize_from_u64("chunk_index_length", sb.chunk_index_length)?;
    let idx_end = idx_start
        .checked_add(idx_len)
        .filter(|&e| e <= data.len())
        .ok_or(CatalogError::TooShort {
            need: idx_start.saturating_add(idx_len),
            got: data.len(),
        })?;
    let idx_bytes = &data[idx_start..idx_end];
    let chunks = index::parse_chunk_index(idx_bytes)?;

    index::validate_chunk_payloads(&chunks, data.len() as u64)?;

    Ok(TetFileSummaryV1 {
        superblock: sb,
        datasets,
        chunks,
    })
}

/// Write a `.tet` with one dataset and any number of **raw** (`codec = 0`) `f32` chunks.
///
/// `data` must be the full tensor in **row-major** order (`4 * product(shape)` bytes).
///
/// # Errors
///
/// Returns I/O errors from the host, or [`CatalogError`] when arguments are inconsistent.
pub fn write_raw_array_file(path: &Path, spec: &RawArrayWrite<'_>) -> Result<(), CatalogError> {
    dataset::validate_raw_array_write(spec)?;
    write_raw_array_file_inner(path, spec)
}

fn write_raw_array_file_inner(path: &Path, spec: &RawArrayWrite<'_>) -> Result<(), CatalogError> {
    let blob = dataset::encode_dataset_blob(spec.name, spec.dtype, spec.shape, spec.chunk_shape)?;
    let dataset_blob_len = blob.len() as u64;
    let index_base = wire::align8_u64(40u64 + dataset_blob_len);

    let ndim = spec.shape.len();
    let counts = tile::chunk_grid_counts(spec.shape, spec.chunk_shape);
    let n_chunks = tile::total_chunk_count(&counts)?;
    let n_usize = usize::try_from(n_chunks).map_err(|_| CatalogError::TooLargeForPlatform {
        field: "chunk_entry_count",
        value: n_chunks,
    })?;

    let index_header_len = index::CHUNK_INDEX_HEADER_V1.header_len as u64;
    let entries_len_u64 = n_chunks
        .checked_mul(ChunkIndexEntryV1::WIRE_LEN as u64)
        .ok_or(CatalogError::InvalidWriteSpec("chunk index size overflow"))?;
    let chunk_index_length = index_header_len
        .checked_add(entries_len_u64)
        .ok_or(CatalogError::InvalidWriteSpec("chunk index size overflow"))?;

    let payload_start = index_base + chunk_index_length;

    let mut entries: Vec<ChunkIndexEntryV1> = Vec::with_capacity(n_usize);
    let mut payloads: Vec<Vec<u8>> = Vec::with_capacity(n_usize);
    let mut cursor = payload_start;

    for k in 0..n_chunks {
        let coord = tile::chunk_coord_from_linear(k, &counts, ndim);
        let tile_bytes = tile::extract_f32_tile_row_major(
            spec.data,
            spec.shape,
            spec.chunk_shape,
            &coord[..ndim],
            ndim,
        )?;
        let raw_len =
            u64::try_from(tile_bytes.len()).map_err(|_| CatalogError::TooLargeForPlatform {
                field: "chunk_payload_len",
                value: u64::MAX,
            })?;
        let mut chunk_index = [0u64; MAX_NDIM];
        chunk_index[..ndim].copy_from_slice(&coord[..ndim]);
        entries.push(ChunkIndexEntryV1 {
            dataset_id: 0,
            chunk_index,
            payload_offset: cursor,
            raw_byte_len: raw_len,
            stored_byte_len: raw_len,
            codec: 0,
        });
        cursor = cursor
            .checked_add(raw_len)
            .ok_or(CatalogError::InvalidWriteSpec("payload cursor overflow"))?;
        payloads.push(tile_bytes);
    }

    let sb = SuperblockV1 {
        layout_version: LAYOUT_VERSION_V1,
        dataset_count: 1,
        flags: 0,
        chunk_index_offset: index_base,
        chunk_index_length,
    };

    let mut index_bytes = Vec::with_capacity(usize_from_u64(
        "chunk_index_byte_length",
        chunk_index_length,
    )?);
    let idx_hdr = index::CHUNK_INDEX_HEADER_V1;
    index_bytes.extend_from_slice(idx_hdr.magic);
    index_bytes.extend_from_slice(&idx_hdr.version.to_le_bytes());
    index_bytes.extend_from_slice(&n_chunks.to_le_bytes());
    index_bytes.resize(idx_hdr.header_len, 0);
    for e in &entries {
        index_bytes.extend_from_slice(&e.to_bytes());
    }

    debug_assert_eq!(index_bytes.len() as u64, chunk_index_length);

    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    f.write_all(&sb.to_bytes())?;
    f.write_all(&dataset_blob_len.to_le_bytes())?;
    f.write_all(&blob)?;
    let after_blob = 40usize + blob.len();
    let pad = usize_from_u64("chunk_index_base", index_base)?.saturating_sub(after_blob);
    if pad > 0 {
        f.write_all(&vec![0u8; pad])?;
    }
    f.write_all(&index_bytes)?;
    for p in payloads {
        f.write_all(&p)?;
    }
    f.sync_all()?;
    Ok(())
}

/// Write a `.tet` containing one dataset and exactly one uncompressed chunk (`codec = 0`).
///
/// # Errors
///
/// Returns I/O errors from the host, or [`CatalogError`] when arguments are inconsistent.
pub fn write_one_chunk_raw_file(
    path: &Path,
    spec: &OneChunkRawWrite<'_>,
) -> Result<(), CatalogError> {
    dataset::validate_write_spec(spec)?;
    write_raw_array_file_inner(
        path,
        &RawArrayWrite {
            name: spec.name,
            dtype: spec.dtype,
            shape: spec.shape,
            chunk_shape: spec.chunk_shape,
            data: spec.payload,
        },
    )
}

fn usize_from_u64(field: &'static str, v: u64) -> Result<usize, CatalogError> {
    usize::try_from(v).map_err(|_| CatalogError::TooLargeForPlatform { field, value: v })
}
