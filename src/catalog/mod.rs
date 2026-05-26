//! Dataset directory and chunk index (layout v1 extension).
//!
//! See `docs/layout_v1.md` for byte layout after the 32-byte superblock.

mod dataset;
pub mod execution;
mod history;
mod index;
pub mod metadata;
pub mod session;
mod stream_write;
pub mod tile;
mod write;

use std::borrow::Cow;
use std::io;

use serde::Serialize;
use thiserror::Error;

use crate::layout::{self, SuperblockV1};
use crate::utils::wire;

pub use dataset::{DatasetRecordV1, RawArrayWrite};
pub use execution::{DEFAULT_MEMORY_BUDGET_PERCENT_BPS, FileExecutionSettingsV1};
pub use history::{
    FooterBlobV1, HistoryEventV1, HistoryFooterWireV1, append_convert_history,
    append_history_events, read_footer_blob, read_metadata, unix_timestamp_now, write_footer_blob,
};
pub use index::{CHUNK_INDEX_HEADER_V1, ChunkIndexEntryV1, ChunkIndexHeaderV1};
pub use metadata::{DatasetMetadataV1, FileMetadataV1, MetadataLimitsV1, TetMetadataV1};
pub use session::{FileMetadataDraft, TetDatasetWrite, TetFile, TetWriterSession};
pub use stream_write::{
    ArrayWriteMeta, StreamTileJob, StreamWriteProgress, total_chunk_count_for_meta,
    validate_array_write_meta, write_multi_raw_array_streaming,
};
pub use tile::{chunk_coords_intersecting_global_box, chunk_coords_intersecting_strided};
pub use write::{write_multi_raw_array_file, write_one_chunk_raw_file, write_raw_array_file};

/// v1 chunk payload codec wire tags (`u32` values written per chunk in the index).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkPayloadCodecV1 {
    /// Raw little-endian tensor bytes (`stored_byte_len == raw_byte_len`).
    pub raw: u32,
    /// **zstd**–compressed bytes at `payload_offset`; decompressed size is `raw_byte_len`.
    pub zstd: u32,
}

/// Defined chunk payload codecs for layout v1 (see `docs/layout_v1.md`).
pub const CHUNK_PAYLOAD_CODEC_V1: ChunkPayloadCodecV1 = ChunkPayloadCodecV1 { raw: 0, zstd: 1 };

impl ChunkPayloadCodecV1 {
    #[must_use]
    pub const fn is_raw(self, codec: u32) -> bool {
        codec == self.raw
    }

    #[must_use]
    pub const fn is_zstd(self, codec: u32) -> bool {
        codec == self.zstd
    }

    #[must_use]
    pub const fn is_supported(self, codec: u32) -> bool {
        self.is_raw(codec) || self.is_zstd(codec)
    }

    /// Encode one tile’s uncompressed `f32` row-major bytes for the given wire `codec`.
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError::UnsupportedCodec`] when `codec` is neither [`ChunkPayloadCodecV1::raw`]
    /// nor [`ChunkPayloadCodecV1::zstd`] for this table, and [`CatalogError::Io`] when zstd
    /// compression fails.
    pub fn encode_tile_payload(
        self,
        codec: u32,
        tile_bytes: Vec<u8>,
    ) -> Result<Vec<u8>, CatalogError> {
        if self.is_raw(codec) {
            return Ok(tile_bytes);
        }
        if self.is_zstd(codec) {
            return zstd::encode_all(tile_bytes.as_slice(), 0).map_err(CatalogError::Io);
        }
        Err(CatalogError::UnsupportedCodec { codec })
    }

    /// Decode stored chunk bytes to uncompressed tile payload (`raw_byte_len` bytes).
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError::UnsupportedCodec`], [`CatalogError::RawStoredMismatch`],
    /// [`CatalogError::ZstdDecode`], or [`CatalogError::DecodedLengthMismatch`].
    pub fn decode_tile_payload(
        self,
        stored: &[u8],
        raw_byte_len: u64,
        stored_byte_len: u64,
        codec: u32,
    ) -> Result<Cow<'_, [u8]>, CatalogError> {
        if self.is_raw(codec) {
            if stored_byte_len != raw_byte_len {
                return Err(CatalogError::RawStoredMismatch {
                    raw: raw_byte_len,
                    stored: stored_byte_len,
                });
            }
            return Ok(Cow::Borrowed(stored));
        }
        if self.is_zstd(codec) {
            let dec =
                zstd::decode_all(stored).map_err(|e| CatalogError::ZstdDecode(e.to_string()))?;
            if dec.len() as u64 != raw_byte_len {
                return Err(CatalogError::DecodedLengthMismatch {
                    decoded: dec.len(),
                    raw: raw_byte_len,
                });
            }
            return Ok(Cow::Owned(dec));
        }
        Err(CatalogError::UnsupportedCodec { codec })
    }
}

/// v1 dataset element `dtype` wire tags (`u32` in each dataset directory record).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DatasetDtypeTagV1 {
    /// IEEE754 binary32 (`f32`), row-major within each chunk.
    pub f32: u32,
    /// IEEE754 binary64 (`f64`), row-major within each chunk.
    pub f64: u32,
    /// Two's-complement `i32`, row-major within each chunk.
    pub i32: u32,
    /// Two's-complement `i64`, row-major within each chunk.
    pub i64: u32,
}

/// Defined dataset element dtypes for layout v1 (see `docs/layout_v1.md`).
pub const DATASET_DTYPE_TAG_V1: DatasetDtypeTagV1 = DatasetDtypeTagV1 {
    f32: 1,
    f64: 2,
    i32: 3,
    i64: 4,
};

impl DatasetDtypeTagV1 {
    #[must_use]
    pub const fn is_f32(self, dtype: u32) -> bool {
        dtype == self.f32
    }

    #[must_use]
    pub const fn is_f64(self, dtype: u32) -> bool {
        dtype == self.f64
    }

    #[must_use]
    pub const fn is_i32(self, dtype: u32) -> bool {
        dtype == self.i32
    }

    #[must_use]
    pub const fn is_i64(self, dtype: u32) -> bool {
        dtype == self.i64
    }

    #[must_use]
    pub const fn is_supported(self, dtype: u32) -> bool {
        self.is_f32(dtype) || self.is_f64(dtype) || self.is_i32(dtype) || self.is_i64(dtype)
    }
}

/// Maximum tensor rank supported by the v1 catalog on disk.
pub const MAX_NDIM: usize = 8;

/// Element count × element size for a tensor with `shape` and wire `dtype`.
#[must_use]
pub fn tensor_bytes_from_shape(shape: &[u64], dtype: u32) -> Option<u64> {
    crate::utils::dtype::ElementDtype::try_from_wire_tag(dtype)?.tensor_bytes_for_shape(shape)
}

/// Typed helpers for callers that already know the element type.
#[must_use]
pub fn f32_tensor_bytes_from_shape(shape: &[u64]) -> Option<u64> {
    crate::utils::dtype::ElementDtype::F32.tensor_bytes_for_shape(shape)
}

/// Element count × 8 for an `f64` tensor with `shape`.
#[must_use]
pub fn f64_tensor_bytes_from_shape(shape: &[u64]) -> Option<u64> {
    crate::utils::dtype::ElementDtype::F64.tensor_bytes_for_shape(shape)
}

/// Element count × 4 for an `i32` tensor with `shape`.
#[must_use]
pub fn i32_tensor_bytes_from_shape(shape: &[u64]) -> Option<u64> {
    crate::utils::dtype::ElementDtype::I32.tensor_bytes_for_shape(shape)
}

/// Element count × 8 for an `i64` tensor with `shape`.
#[must_use]
pub fn i64_tensor_bytes_from_shape(shape: &[u64]) -> Option<u64> {
    crate::utils::dtype::ElementDtype::I64.tensor_bytes_for_shape(shape)
}

/// High-level view of a mapped `.tet` file (superblock + catalog).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct TetFileSummaryV1 {
    pub superblock: SuperblockV1,
    pub datasets: Vec<DatasetRecordV1>,
    pub chunks: Vec<ChunkIndexEntryV1>,
    /// Execution preferences from the chunk index header (defaults when all zero).
    pub file_execution: FileExecutionSettingsV1,
    /// Optional provenance/history footer (`[["convert","h5"|"nc","<unix_secs>"], …]`).
    pub history: Vec<HistoryEventV1>,
    /// Optional `metadata` object from the same `THST` footer JSON.
    pub metadata: TetMetadataV1,
}

/// Catalog read, index validation, codec, and writer failures.
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
    #[error("unsupported codec {codec} (supported: 0 = raw, 1 = zstd)")]
    UnsupportedCodec { codec: u32 },
    #[error("raw/stored length mismatch for codec=0: raw={raw}, stored={stored}")]
    RawStoredMismatch { raw: u64, stored: u64 },
    #[error("zstd decode failed: {0}")]
    ZstdDecode(String),
    #[error("decoded payload length {decoded} != raw_byte_len {raw}")]
    DecodedLengthMismatch { decoded: usize, raw: u64 },
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
        let flags = sb.flags;
        return Ok(TetFileSummaryV1 {
            superblock: sb,
            datasets: Vec::new(),
            chunks: Vec::new(),
            file_execution: FileExecutionSettingsV1::default_engine(),
            history: history::read_history(data, flags)?,
            metadata: history::read_metadata(data, flags)?,
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
    let (chunks, file_execution) = index::parse_chunk_index(idx_bytes)?;

    let payload_len = history::payload_file_len(data, sb.flags)?;
    index::validate_chunk_payloads(&chunks, payload_len)?;
    let history = history::read_history(data, sb.flags)?;
    let metadata = history::read_metadata(data, sb.flags)?;

    Ok(TetFileSummaryV1 {
        superblock: sb,
        datasets,
        chunks,
        file_execution,
        history,
        metadata,
    })
}

/// Validate chunk index payload spans against a file length (same rules as [`read_tet_summary_v1`]).
///
/// Exposed for integration tests in [`crate::tests`].
pub use index::validate_chunk_payloads;

pub(super) fn usize_from_u64(field: &'static str, v: u64) -> Result<usize, CatalogError> {
    usize::try_from(v).map_err(|_| CatalogError::TooLargeForPlatform { field, value: v })
}
