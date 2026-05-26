//! Append datasets to an existing layout v1 `.tet` (rewrites catalog, index, and payloads).

use std::collections::HashSet;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

use crate::layout::{LAYOUT_VERSION_V1, SuperblockV1};
use crate::utils::wire;

use super::dataset::{self, RawArrayWrite};
use super::execution::FileExecutionSettingsV1;
use super::index::{self, ChunkIndexEntryV1};
use super::stream_write::{ArrayWriteMeta, StreamTileJob};
use super::tile;
use super::{
    CHUNK_PAYLOAD_CODEC_V1, CatalogError, DATASET_DTYPE_TAG_V1, DatasetRecordV1, MAX_NDIM,
    read_tet_summary_v1, usize_from_u64,
};
use crate::utils::dtype::ElementDtype;

/// Append in-memory datasets to an existing file (rewrites the file; strips an EOF footer until
/// the caller rewrites it).
///
/// Existing chunk bytes are copied verbatim. New dataset names must be unique.
///
/// # Errors
///
/// Returns [`CatalogError`] when the file is invalid, a name collides, or I/O fails.
pub fn append_multi_raw_array_file(
    path: &Path,
    new_specs: &[RawArrayWrite<'_>],
) -> Result<(), CatalogError> {
    if new_specs.is_empty() {
        return Ok(());
    }
    for spec in new_specs {
        dataset::validate_raw_array_write(spec)?;
    }
    let data = std::fs::read(path)?;
    let summary = read_tet_summary_v1(&data)?;
    ensure_new_names(&summary.datasets, new_specs)?;

    let file_execution = new_specs[0]
        .file_execution
        .unwrap_or(summary.file_execution);
    let (entries, payloads) = copy_existing_chunks(&data, &summary.chunks)?;
    let (entries, payloads) =
        append_in_memory_chunks(entries, payloads, summary.datasets.len(), new_specs)?;
    rewrite_file(
        path,
        &summary.datasets,
        new_specs,
        &[],
        entries,
        &payloads,
        file_execution,
    )
}

/// Append in-memory and streaming datasets in one rewrite (raw codec for streaming tiles).
///
/// # Errors
///
/// Returns [`CatalogError`] when the file is invalid, `fill` fails, or I/O fails.
pub fn append_multi_mixed(
    path: &Path,
    new_specs: &[RawArrayWrite<'_>],
    new_streaming: &[ArrayWriteMeta<'_>],
    fill: impl Fn(&StreamTileJob<'_>, &mut [u8]) -> Result<(), CatalogError> + Sync + Send,
) -> Result<(), CatalogError> {
    if new_specs.is_empty() && new_streaming.is_empty() {
        return Ok(());
    }
    for spec in new_specs {
        dataset::validate_raw_array_write(spec)?;
    }
    for spec in new_streaming {
        super::validate_array_write_meta(spec)?;
    }
    let data = std::fs::read(path)?;
    let summary = read_tet_summary_v1(&data)?;
    ensure_new_names(&summary.datasets, new_specs)?;
    ensure_new_meta_names(&summary.datasets, new_streaming)?;
    ensure_no_cross_name_duplicates(new_specs, new_streaming)?;

    let file_execution = new_specs
        .first()
        .and_then(|s| s.file_execution)
        .unwrap_or(summary.file_execution);

    let (entries, payloads) = copy_existing_chunks(&data, &summary.chunks)?;
    let base = summary.datasets.len();
    let (entries, payloads) = append_in_memory_chunks(entries, payloads, base, new_specs)?;
    let (entries, payloads) = append_streaming_chunks(
        entries,
        payloads,
        base + new_specs.len(),
        new_streaming,
        &fill,
    )?;
    rewrite_file(
        path,
        &summary.datasets,
        new_specs,
        new_streaming,
        entries,
        &payloads,
        file_execution,
    )
}

/// Append streaming datasets (raw codec only) to an existing file.
///
/// # Errors
///
/// Returns [`CatalogError`] when the file is invalid, `fill` fails, or I/O fails.
pub fn append_multi_raw_array_streaming(
    path: &Path,
    new_specs: &[ArrayWriteMeta<'_>],
    fill: impl Fn(&StreamTileJob<'_>, &mut [u8]) -> Result<(), CatalogError> + Sync + Send,
) -> Result<(), CatalogError> {
    if new_specs.is_empty() {
        return Ok(());
    }
    for spec in new_specs {
        super::validate_array_write_meta(spec)?;
    }
    let data = std::fs::read(path)?;
    let summary = read_tet_summary_v1(&data)?;
    ensure_new_meta_names(&summary.datasets, new_specs)?;

    let (entries, payloads) = copy_existing_chunks(&data, &summary.chunks)?;
    let (entries, payloads) =
        append_streaming_chunks(entries, payloads, summary.datasets.len(), new_specs, &fill)?;
    rewrite_file(
        path,
        &summary.datasets,
        &[],
        new_specs,
        entries,
        &payloads,
        summary.file_execution,
    )
}

fn ensure_new_names(
    existing: &[DatasetRecordV1],
    new_specs: &[RawArrayWrite<'_>],
) -> Result<(), CatalogError> {
    let mut seen: HashSet<&str> = existing.iter().map(|d| d.name.as_str()).collect();
    for spec in new_specs {
        if !seen.insert(spec.name) {
            return Err(CatalogError::InvalidWriteSpec(
                "dataset name already exists in file",
            ));
        }
    }
    Ok(())
}

fn ensure_no_cross_name_duplicates(
    new_specs: &[RawArrayWrite<'_>],
    new_streaming: &[ArrayWriteMeta<'_>],
) -> Result<(), CatalogError> {
    let mut seen: HashSet<&str> = new_specs.iter().map(|s| s.name).collect();
    for spec in new_streaming {
        if !seen.insert(spec.name) {
            return Err(CatalogError::InvalidWriteSpec(
                "duplicate dataset name in append batch",
            ));
        }
    }
    Ok(())
}

fn ensure_new_meta_names(
    existing: &[DatasetRecordV1],
    new_specs: &[ArrayWriteMeta<'_>],
) -> Result<(), CatalogError> {
    let mut seen: HashSet<&str> = existing.iter().map(|d| d.name.as_str()).collect();
    for spec in new_specs {
        if !seen.insert(spec.name) {
            return Err(CatalogError::InvalidWriteSpec(
                "dataset name already exists in file",
            ));
        }
    }
    Ok(())
}

fn copy_stored_payload(data: &[u8], entry: &ChunkIndexEntryV1) -> Result<Vec<u8>, CatalogError> {
    let start = usize_from_u64("payload_offset", entry.payload_offset)?;
    let len = usize_from_u64("stored_byte_len", entry.stored_byte_len)?;
    let end = start
        .checked_add(len)
        .ok_or(CatalogError::InvalidWriteSpec("payload span overflow"))?;
    if end > data.len() {
        return Err(CatalogError::PayloadOutOfBounds {
            file_len: data.len() as u64,
            start: entry.payload_offset,
            end: entry.payload_offset.saturating_add(entry.stored_byte_len),
        });
    }
    Ok(data[start..end].to_vec())
}

fn copy_existing_chunks(
    data: &[u8],
    existing_chunks: &[ChunkIndexEntryV1],
) -> Result<(Vec<ChunkIndexEntryV1>, Vec<Vec<u8>>), CatalogError> {
    let mut entries = Vec::with_capacity(existing_chunks.len());
    let mut payloads = Vec::with_capacity(existing_chunks.len());
    let mut cursor = 0u64;
    for entry in existing_chunks {
        let stored = copy_stored_payload(data, entry)?;
        cursor = push_entry(&mut entries, &mut payloads, entry, cursor, stored)?;
    }
    Ok((entries, payloads))
}

fn payload_cursor(entries: &[ChunkIndexEntryV1]) -> u64 {
    entries
        .last()
        .map(|e| e.payload_offset.saturating_add(e.stored_byte_len))
        .unwrap_or(0)
}

fn push_entry(
    entries: &mut Vec<ChunkIndexEntryV1>,
    payloads: &mut Vec<Vec<u8>>,
    template: &ChunkIndexEntryV1,
    cursor: u64,
    stored: Vec<u8>,
) -> Result<u64, CatalogError> {
    let stored_len =
        u64::try_from(stored.len()).map_err(|_| CatalogError::TooLargeForPlatform {
            field: "chunk_stored_len",
            value: u64::MAX,
        })?;
    entries.push(ChunkIndexEntryV1 {
        dataset_id: template.dataset_id,
        chunk_index: template.chunk_index,
        payload_offset: cursor,
        raw_byte_len: template.raw_byte_len,
        stored_byte_len: stored_len,
        codec: template.codec,
    });
    let next = cursor
        .checked_add(stored_len)
        .ok_or(CatalogError::InvalidWriteSpec("payload cursor overflow"))?;
    payloads.push(stored);
    Ok(next)
}

fn append_in_memory_chunks(
    mut entries: Vec<ChunkIndexEntryV1>,
    mut payloads: Vec<Vec<u8>>,
    base_dataset_id: usize,
    new_specs: &[RawArrayWrite<'_>],
) -> Result<(Vec<ChunkIndexEntryV1>, Vec<Vec<u8>>), CatalogError> {
    let mut cursor = payload_cursor(&entries);
    let tags = DATASET_DTYPE_TAG_V1;

    for (offset, spec) in new_specs.iter().enumerate() {
        let dataset_id = base_dataset_id + offset;
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
            let mut chunk_index = [0u64; MAX_NDIM];
            chunk_index[..ndim].copy_from_slice(&coord[..ndim]);
            let template = ChunkIndexEntryV1 {
                dataset_id: u64::try_from(dataset_id).map_err(|_| {
                    CatalogError::TooLargeForPlatform {
                        field: "dataset_id",
                        value: u64::MAX,
                    }
                })?,
                chunk_index,
                payload_offset: 0,
                raw_byte_len: raw_len,
                stored_byte_len: 0,
                codec: spec.chunk_codec,
            };
            cursor = push_entry(&mut entries, &mut payloads, &template, cursor, stored_vec)?;
        }
    }
    Ok((entries, payloads))
}

fn append_streaming_chunks<F>(
    mut entries: Vec<ChunkIndexEntryV1>,
    mut payloads: Vec<Vec<u8>>,
    base_dataset_id: usize,
    new_specs: &[ArrayWriteMeta<'_>],
    fill: &F,
) -> Result<(Vec<ChunkIndexEntryV1>, Vec<Vec<u8>>), CatalogError>
where
    F: Fn(&StreamTileJob<'_>, &mut [u8]) -> Result<(), CatalogError>,
{
    let mut cursor = payload_cursor(&entries);

    for (offset, spec) in new_specs.iter().enumerate() {
        let dataset_id = base_dataset_id + offset;
        let ndim = spec.shape.len();
        let counts = tile::chunk_grid_counts(spec.shape, spec.chunk_shape);
        let n_chunks = tile::total_chunk_count(&counts)?;
        let elem_size = element_size_for_dtype(spec.dtype)?;

        for k in 0..n_chunks {
            let coord = tile::chunk_coord_from_linear(k, &counts, ndim);
            let raw_len = tile::tile_raw_byte_len(
                spec.shape,
                spec.chunk_shape,
                &coord[..ndim],
                ndim,
                elem_size,
            )?;
            let mut tile_buf = vec![0u8; usize_from_u64("tile_raw_byte_len", raw_len)?];
            let mut chunk_index = [0u64; MAX_NDIM];
            chunk_index[..ndim].copy_from_slice(&coord[..ndim]);
            let job = StreamTileJob {
                dataset_id,
                dataset_name: spec.name,
                chunk_k: k,
                chunk_coord: chunk_index,
                ndim,
                raw_byte_len: raw_len,
            };
            fill(&job, &mut tile_buf)?;
            let stored_vec =
                CHUNK_PAYLOAD_CODEC_V1.encode_tile_payload(spec.chunk_codec, tile_buf)?;
            let template = ChunkIndexEntryV1 {
                dataset_id: u64::try_from(dataset_id).map_err(|_| {
                    CatalogError::TooLargeForPlatform {
                        field: "dataset_id",
                        value: u64::MAX,
                    }
                })?,
                chunk_index,
                payload_offset: 0,
                raw_byte_len: raw_len,
                stored_byte_len: 0,
                codec: spec.chunk_codec,
            };
            cursor = push_entry(&mut entries, &mut payloads, &template, cursor, stored_vec)?;
        }
    }
    Ok((entries, payloads))
}

fn rewrite_file(
    path: &Path,
    existing_datasets: &[DatasetRecordV1],
    new_specs: &[RawArrayWrite<'_>],
    new_streaming: &[ArrayWriteMeta<'_>],
    mut entries: Vec<ChunkIndexEntryV1>,
    payloads: &[Vec<u8>],
    file_execution: FileExecutionSettingsV1,
) -> Result<(), CatalogError> {
    let mut blob = Vec::new();
    for ds in existing_datasets {
        blob.extend_from_slice(&dataset::encode_dataset_blob(
            &ds.name,
            ds.dtype,
            &ds.shape,
            &ds.chunk_shape,
        )?);
    }
    for spec in new_specs {
        blob.extend_from_slice(&dataset::encode_dataset_blob(
            spec.name,
            spec.dtype,
            spec.shape,
            spec.chunk_shape,
        )?);
    }
    for spec in new_streaming {
        blob.extend_from_slice(&dataset::encode_dataset_blob(
            spec.name,
            spec.dtype,
            spec.shape,
            spec.chunk_shape,
        )?);
    }

    let dataset_count = existing_datasets
        .len()
        .checked_add(new_specs.len())
        .and_then(|n| n.checked_add(new_streaming.len()))
        .ok_or(CatalogError::InvalidWriteSpec("dataset count overflow"))?;
    let dataset_blob_len = blob.len() as u64;
    let index_base = wire::align8_u64(40u64 + dataset_blob_len);
    let n_chunks_total =
        u64::try_from(entries.len()).map_err(|_| CatalogError::TooLargeForPlatform {
            field: "chunk_entry_count",
            value: u64::MAX,
        })?;
    let index_header_len = index::CHUNK_INDEX_HEADER_V1.header_len as u64;
    let entries_len_u64 = n_chunks_total
        .checked_mul(ChunkIndexEntryV1::WIRE_LEN as u64)
        .ok_or(CatalogError::InvalidWriteSpec("chunk index size overflow"))?;
    let chunk_index_length = index_header_len
        .checked_add(entries_len_u64)
        .ok_or(CatalogError::InvalidWriteSpec("chunk index size overflow"))?;

    let payload_start = index_base
        .checked_add(chunk_index_length)
        .ok_or(CatalogError::InvalidWriteSpec("payload start overflow"))?;
    let mut cursor = payload_start;
    for entry in &mut entries {
        entry.payload_offset = cursor;
        cursor = cursor
            .checked_add(entry.stored_byte_len)
            .ok_or(CatalogError::InvalidWriteSpec("payload cursor overflow"))?;
    }

    let sb = SuperblockV1 {
        layout_version: LAYOUT_VERSION_V1,
        dataset_count: u32::try_from(dataset_count).map_err(|_| {
            CatalogError::TooLargeForPlatform {
                field: "dataset_count",
                value: dataset_count as u64,
            }
        })?,
        flags: 0,
        chunk_index_offset: index_base,
        chunk_index_length,
    };

    let mut index_bytes = Vec::with_capacity(usize_from_u64(
        "chunk_index_byte_length",
        chunk_index_length,
    )?);
    index::write_chunk_index_header(&mut index_bytes, n_chunks_total, file_execution);
    for e in entries {
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
        f.write_all(p)?;
    }
    f.sync_all()?;
    Ok(())
}

fn element_size_for_dtype(dtype: u32) -> Result<usize, CatalogError> {
    let elem = ElementDtype::try_from_wire_tag(dtype).ok_or(CatalogError::InvalidWriteSpec(
        "unsupported dataset dtype tag",
    ))?;
    Ok(elem.elem_size())
}
