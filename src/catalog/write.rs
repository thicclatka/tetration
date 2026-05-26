//! Write paths for layout v1 `.tet` files.
//!
//! Multi-dataset creates delegate layout math and tile iteration to [`crate::catalog::file_layout`].

use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

use super::dataset::{self, RawArrayWrite};
use super::index::ChunkIndexEntryV1;
use super::{CHUNK_PAYLOAD_CODEC_V1, CatalogError, FileExecutionSettingsV1, OneChunkRawWrite};

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
    let n_chunks_total =
        super::file_layout::sum_chunk_counts(specs.iter().map(|s| (s.shape, s.chunk_shape)))?;
    let (index_base, chunk_index_length, payload_start) =
        super::file_layout::chunk_index_layout(dataset_blob_len, n_chunks_total)?;

    let mut entries: Vec<ChunkIndexEntryV1> = Vec::new();
    let mut payloads: Vec<Vec<u8>> = Vec::new();
    let mut cursor = payload_start;
    for (dataset_id, spec) in specs.iter().enumerate() {
        let grid = super::chunk_grid_plan(spec.shape, spec.chunk_shape, spec.dtype)?;
        cursor = super::file_layout::push_raw_tiles_from_tensor(
            spec,
            &grid,
            super::file_layout::wire_dataset_id(dataset_id)?,
            cursor,
            &mut entries,
            &mut payloads,
        )?;
    }

    let sb = super::file_layout::layout_superblock(specs.len(), index_base, chunk_index_length)?;
    let index_bytes = super::file_layout::build_chunk_index_bytes(
        &entries,
        n_chunks_total,
        chunk_index_length,
        specs[0]
            .file_execution
            .unwrap_or_else(FileExecutionSettingsV1::default_engine),
    )?;

    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    super::file_layout::write_file_preamble(&mut f, &sb, &blob, index_base, &index_bytes)?;
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
