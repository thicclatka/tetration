//! Write paths for layout v1 `.tet` files.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

use crate::layout::{LAYOUT_VERSION_V1, SuperblockV1};
use crate::utils::wire;

use super::dataset::{self, RawArrayWrite};
use super::index::{self, ChunkIndexEntryV1};
use super::tile;
use super::{
    CHUNK_PAYLOAD_CODEC_V1, CatalogError, DATASET_DTYPE_TAG_V1, FileExecutionSettingsV1, MAX_NDIM,
    OneChunkRawWrite,
};
use crate::utils::dtype::ElementDtype;

/// Write a `.tet` with one dataset and any number of chunks (row-major `f32` or `f64` elements),
/// each stored as [`CHUNK_PAYLOAD_CODEC_V1`].[`raw`](super::ChunkPayloadCodecV1::raw) or
/// [`zstd`](super::ChunkPayloadCodecV1::zstd) per `spec.chunk_codec`.
///
/// `data` must be the full tensor in **row-major** order (`element_size * product(shape)` bytes).
///
/// # Errors
///
/// Returns I/O errors from the host, or [`CatalogError`] when arguments are inconsistent.
pub fn write_raw_array_file(path: &Path, spec: &RawArrayWrite<'_>) -> Result<(), CatalogError> {
    write_multi_raw_array_file(path, std::slice::from_ref(spec))
}

/// Write a `.tet` with multiple datasets and tiled payloads (one chunk index shared by all datasets).
///
/// # Errors
///
/// Returns I/O errors from the host, or [`CatalogError`] when arguments are inconsistent.
pub fn write_multi_raw_array_file(
    path: &Path,
    specs: &[RawArrayWrite<'_>],
) -> Result<(), CatalogError> {
    if specs.is_empty() {
        return Err(CatalogError::InvalidWriteSpec(
            "at least one dataset is required",
        ));
    }
    for spec in specs {
        dataset::validate_raw_array_write(spec)?;
    }
    write_multi_raw_array_file_inner(path, specs)
}

#[allow(clippy::too_many_lines)]
fn write_multi_raw_array_file_inner(
    path: &Path,
    specs: &[RawArrayWrite<'_>],
) -> Result<(), CatalogError> {
    let mut blob = Vec::new();
    for spec in specs {
        blob.extend_from_slice(&dataset::encode_dataset_blob(
            spec.name,
            spec.dtype,
            spec.shape,
            spec.chunk_shape,
        )?);
    }
    let dataset_blob_len = blob.len() as u64;
    let index_base = wire::align8_u64(40u64 + dataset_blob_len);

    let mut entries: Vec<ChunkIndexEntryV1> = Vec::new();
    let mut payloads: Vec<Vec<u8>> = Vec::new();

    let index_header_len = index::CHUNK_INDEX_HEADER_V1.header_len as u64;
    let mut n_chunks_total: u64 = 0;
    for spec in specs {
        let counts = tile::chunk_grid_counts(spec.shape, spec.chunk_shape);
        n_chunks_total = n_chunks_total
            .checked_add(tile::total_chunk_count(&counts)?)
            .ok_or(CatalogError::InvalidWriteSpec("chunk index size overflow"))?;
    }
    let entries_len_u64 = n_chunks_total
        .checked_mul(ChunkIndexEntryV1::WIRE_LEN as u64)
        .ok_or(CatalogError::InvalidWriteSpec("chunk index size overflow"))?;
    let chunk_index_length = index_header_len
        .checked_add(entries_len_u64)
        .ok_or(CatalogError::InvalidWriteSpec("chunk index size overflow"))?;

    let payload_start = index_base + chunk_index_length;
    let mut cursor = payload_start;

    let tags = DATASET_DTYPE_TAG_V1;
    for (dataset_id, spec) in specs.iter().enumerate() {
        let ndim = spec.shape.len();
        let counts = tile::chunk_grid_counts(spec.shape, spec.chunk_shape);
        let n_chunks = tile::total_chunk_count(&counts)?;
        let elem_size = if tags.is_f32(spec.dtype) {
            ElementDtype::F32.elem_size()
        } else if tags.is_f64(spec.dtype) {
            ElementDtype::F64.elem_size()
        } else if tags.is_i32(spec.dtype) {
            ElementDtype::I32.elem_size()
        } else if tags.is_i64(spec.dtype) {
            ElementDtype::I64.elem_size()
        } else {
            return Err(CatalogError::InvalidWriteSpec(
                "unsupported dtype for tile extraction",
            ));
        };

        for k in 0..n_chunks {
            let coord = tile::chunk_coord_from_linear(k, &counts, ndim);
            let tile_bytes = tile::extract_tile_row_major(
                spec.data,
                spec.shape,
                spec.chunk_shape,
                &coord[..ndim],
                ndim,
                elem_size,
            )?;
            let raw_len =
                u64::try_from(tile_bytes.len()).map_err(|_| CatalogError::TooLargeForPlatform {
                    field: "chunk_payload_len",
                    value: u64::MAX,
                })?;
            let stored_vec =
                CHUNK_PAYLOAD_CODEC_V1.encode_tile_payload(spec.chunk_codec, tile_bytes)?;
            let stored_len =
                u64::try_from(stored_vec.len()).map_err(|_| CatalogError::TooLargeForPlatform {
                    field: "chunk_stored_len",
                    value: u64::MAX,
                })?;
            let mut chunk_index = [0u64; MAX_NDIM];
            chunk_index[..ndim].copy_from_slice(&coord[..ndim]);
            entries.push(ChunkIndexEntryV1 {
                dataset_id: u64::try_from(dataset_id).map_err(|_| {
                    CatalogError::TooLargeForPlatform {
                        field: "dataset_id",
                        value: u64::MAX,
                    }
                })?,
                chunk_index,
                payload_offset: cursor,
                raw_byte_len: raw_len,
                stored_byte_len: stored_len,
                codec: spec.chunk_codec,
            });
            cursor = cursor
                .checked_add(stored_len)
                .ok_or(CatalogError::InvalidWriteSpec("payload cursor overflow"))?;
            payloads.push(stored_vec);
        }
    }

    let sb = SuperblockV1 {
        layout_version: LAYOUT_VERSION_V1,
        dataset_count: u32::try_from(specs.len()).map_err(|_| {
            CatalogError::TooLargeForPlatform {
                field: "dataset_count",
                value: specs.len() as u64,
            }
        })?,
        flags: 0,
        chunk_index_offset: index_base,
        chunk_index_length,
    };

    let mut index_bytes = Vec::with_capacity(super::usize_from_u64(
        "chunk_index_byte_length",
        chunk_index_length,
    )?);
    index::write_chunk_index_header(
        &mut index_bytes,
        n_chunks_total,
        specs[0]
            .file_execution
            .unwrap_or_else(FileExecutionSettingsV1::default_engine),
    );
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
    let pad = super::usize_from_u64("chunk_index_base", index_base)?.saturating_sub(after_blob);
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

#[allow(clippy::too_many_lines)]
fn write_raw_array_file_inner(path: &Path, spec: &RawArrayWrite<'_>) -> Result<(), CatalogError> {
    write_multi_raw_array_file_inner(path, std::slice::from_ref(spec))
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
            chunk_codec: CHUNK_PAYLOAD_CODEC_V1.raw,
            data: spec.payload,
            file_execution: None,
        },
    )
}
