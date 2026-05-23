//! Memory-efficient `.tet` writer: one chunk payload at a time (raw codec).

use std::fs::OpenOptions;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

use rayon::prelude::*;

use crate::layout::{LAYOUT_VERSION_V1, SuperblockV1};
use crate::utils::wire;

use super::dataset::encode_dataset_blob;
use super::execution::FileExecutionSettingsV1;
use super::index::{self, ChunkIndexEntryV1};
use super::tile;
use super::{CHUNK_PAYLOAD_CODEC_V1, CatalogError, DATASET_DTYPE_TAG_V1, MAX_NDIM, usize_from_u64};
use crate::utils::dtype::ElementDtype;

/// Dataset metadata for streaming writes (no in-memory tensor buffer).
#[derive(Debug, Clone)]
pub struct ArrayWriteMeta<'a> {
    pub name: &'a str,
    pub dtype: u32,
    pub shape: &'a [u64],
    pub chunk_shape: &'a [u64],
    pub chunk_codec: u32,
    pub file_execution: Option<FileExecutionSettingsV1>,
}

/// Progress hook: `(chunks_done, chunks_total, current_dataset_name)`.
pub type StreamWriteProgress<'a> = dyn FnMut(u64, u64, &str) + 'a;

/// Identifies one chunk tile the caller must fill during a streaming write.
#[derive(Debug, Clone)]
pub struct StreamTileJob<'a> {
    pub dataset_id: usize,
    pub dataset_name: &'a str,
    pub chunk_k: u64,
    pub chunk_coord: [u64; MAX_NDIM],
    pub ndim: usize,
    pub raw_byte_len: u64,
}

/// Validate [`ArrayWriteMeta`] (shape/chunk grid/dtype/codec) without requiring tensor bytes.
///
/// # Errors
///
/// Returns [`CatalogError`] when arguments are inconsistent.
pub fn validate_array_write_meta(spec: &ArrayWriteMeta<'_>) -> Result<(), CatalogError> {
    if spec.shape.len() != spec.chunk_shape.len() {
        return Err(CatalogError::InvalidWriteSpec(
            "shape and chunk_shape must have the same rank",
        ));
    }
    let ndim = spec.shape.len();
    if ndim == 0 {
        return Err(CatalogError::InvalidWriteSpec(
            "tensor rank must be at least 1",
        ));
    }
    if ndim > MAX_NDIM {
        return Err(CatalogError::BadNdim { ndim });
    }
    for i in 0..ndim {
        if spec.chunk_shape[i] == 0 || spec.shape[i] == 0 {
            return Err(CatalogError::InvalidWriteSpec(
                "shape and chunk_shape entries must be non-zero",
            ));
        }
    }
    if !DATASET_DTYPE_TAG_V1.is_supported(spec.dtype) {
        return Err(CatalogError::InvalidWriteSpec(
            "only dataset dtype tags in DATASET_DTYPE_TAG_V1 (f32/f64/i32/i64) are supported",
        ));
    }
    let _ = super::tensor_bytes_from_shape(spec.shape, spec.dtype)
        .ok_or(CatalogError::InvalidWriteSpec("payload size overflow"))?;
    let counts = tile::chunk_grid_counts(spec.shape, spec.chunk_shape);
    let _ = tile::total_chunk_count(&counts)?;
    if !CHUNK_PAYLOAD_CODEC_V1.is_raw(spec.chunk_codec) {
        return Err(CatalogError::InvalidWriteSpec(
            "streaming write supports raw chunk codec (0) only",
        ));
    }
    Ok(())
}

/// Total chunk count across all datasets in `specs`.
///
/// # Errors
///
/// Returns [`CatalogError::InvalidWriteSpec`] when a per-dataset chunk grid count overflows or
/// the summed total exceeds `u64::MAX`.
pub fn total_chunk_count_for_meta(specs: &[ArrayWriteMeta<'_>]) -> Result<u64, CatalogError> {
    let mut total = 0u64;
    for spec in specs {
        let counts = tile::chunk_grid_counts(spec.shape, spec.chunk_shape);
        total = total
            .checked_add(tile::total_chunk_count(&counts)?)
            .ok_or(CatalogError::InvalidWriteSpec("chunk index size overflow"))?;
    }
    Ok(total)
}

/// Write a `.tet` by filling one tile at a time via `fill_tile` (peak RAM ≈ one chunk).
///
/// When `parallel_jobs` is greater than 1, reads up to that many chunks concurrently via
/// Rayon, then writes payloads in order (peak RAM ≈ `parallel_jobs` × largest tile).
///
/// `on_progress`, when set, is invoked after each payload is flushed (`chunks_done`, `chunks_total`,
/// current dataset name).
///
/// # Errors
///
/// Returns [`CatalogError`] when layout rules are violated, `fill_tile` fails, or I/O fails.
pub fn write_multi_raw_array_streaming(
    path: &Path,
    specs: &[ArrayWriteMeta<'_>],
    parallel_jobs: usize,
    fill_tile: impl Fn(&StreamTileJob<'_>, &mut [u8]) -> Result<(), CatalogError> + Sync + Send,
    on_progress: Option<&mut StreamWriteProgress<'_>>,
) -> Result<(), CatalogError> {
    if specs.is_empty() {
        return Err(CatalogError::InvalidWriteSpec(
            "at least one dataset is required",
        ));
    }
    for spec in specs {
        validate_array_write_meta(spec)?;
    }

    let blob = encode_all_dataset_blobs(specs)?;
    let dataset_blob_len = blob.len() as u64;
    let n_chunks_total = total_chunk_count_for_meta(specs)?;
    let (index_base, chunk_index_length, payload_start) =
        chunk_index_layout(dataset_blob_len, n_chunks_total)?;
    let (entries, jobs) = build_stream_index_and_jobs(specs, payload_start, n_chunks_total)?;
    let sb = stream_superblock(specs, index_base, chunk_index_length)?;
    let file_execution = specs[0]
        .file_execution
        .unwrap_or_else(FileExecutionSettingsV1::default_engine);
    let index_bytes =
        build_index_bytes(&entries, n_chunks_total, chunk_index_length, file_execution)?;

    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    write_file_preamble(&mut f, &sb, &blob, index_base, &index_bytes)?;
    let parallel_jobs = parallel_jobs.max(1);
    if parallel_jobs == 1 {
        write_filled_tiles_sequential(
            &mut f,
            &entries,
            &jobs,
            n_chunks_total,
            &fill_tile,
            on_progress,
        )?;
    } else {
        write_filled_tiles_parallel(
            &mut f,
            &entries,
            &jobs,
            n_chunks_total,
            parallel_jobs,
            &fill_tile,
            on_progress,
        )?;
    }
    f.sync_all()?;
    Ok(())
}

fn write_filled_tiles_sequential(
    f: &mut std::fs::File,
    entries: &[ChunkIndexEntryV1],
    jobs: &[StreamTileJob<'_>],
    n_chunks_total: u64,
    fill_tile: &(impl Fn(&StreamTileJob<'_>, &mut [u8]) -> Result<(), CatalogError> + Sync + Send),
    mut on_progress: Option<&mut StreamWriteProgress<'_>>,
) -> Result<(), CatalogError> {
    let sequential = entries_are_sequential(entries);
    if sequential && !entries.is_empty() {
        let pos = f.stream_position()?;
        if pos != entries[0].payload_offset {
            f.seek(SeekFrom::Start(entries[0].payload_offset))?;
        }
    }

    let mut tile_buf = Vec::new();
    for (i, job) in jobs.iter().enumerate() {
        let entry = &entries[i];
        prepare_tile_buffer(&mut tile_buf, job.raw_byte_len)?;
        fill_tile(job, &mut tile_buf)?;
        if tile_buf.len() as u64 != entry.raw_byte_len {
            return Err(CatalogError::InvalidWriteSpec(
                "fill_tile returned wrong byte length for chunk",
            ));
        }
        write_tile_payload(f, entry, &tile_buf, sequential)?;
        if let Some(ref mut progress) = on_progress {
            progress(
                u64::try_from(i + 1).unwrap_or(u64::MAX),
                n_chunks_total,
                job.dataset_name,
            );
        }
    }
    Ok(())
}

fn write_filled_tiles_parallel(
    f: &mut std::fs::File,
    entries: &[ChunkIndexEntryV1],
    jobs: &[StreamTileJob<'_>],
    n_chunks_total: u64,
    parallel_jobs: usize,
    fill_tile: &(impl Fn(&StreamTileJob<'_>, &mut [u8]) -> Result<(), CatalogError> + Sync + Send),
    mut on_progress: Option<&mut StreamWriteProgress<'_>>,
) -> Result<(), CatalogError> {
    let sequential = entries_are_sequential(entries);
    if sequential && !entries.is_empty() {
        let pos = f.stream_position()?;
        if pos != entries[0].payload_offset {
            f.seek(SeekFrom::Start(entries[0].payload_offset))?;
        }
    }

    let mut i = 0usize;
    while i < jobs.len() {
        let end = (i + parallel_jobs).min(jobs.len());
        let mut batch: Vec<(usize, Vec<u8>)> = (i..end)
            .into_par_iter()
            .map(|idx| {
                let job = &jobs[idx];
                let mut buf = Vec::new();
                prepare_tile_buffer(&mut buf, job.raw_byte_len)?;
                fill_tile(job, &mut buf)?;
                Ok((idx, buf))
            })
            .collect::<Result<Vec<_>, CatalogError>>()?;
        batch.sort_by_key(|(idx, _)| *idx);
        for (idx, tile_buf) in batch {
            let entry = &entries[idx];
            if tile_buf.len() as u64 != entry.raw_byte_len {
                return Err(CatalogError::InvalidWriteSpec(
                    "fill_tile returned wrong byte length for chunk",
                ));
            }
            write_tile_payload(f, entry, &tile_buf, sequential)?;
            if let Some(ref mut progress) = on_progress {
                progress(
                    u64::try_from(idx + 1).unwrap_or(u64::MAX),
                    n_chunks_total,
                    jobs[idx].dataset_name,
                );
            }
        }
        i = end;
    }
    Ok(())
}

fn write_tile_payload(
    f: &mut std::fs::File,
    entry: &ChunkIndexEntryV1,
    tile_buf: &[u8],
    sequential: bool,
) -> Result<(), CatalogError> {
    if sequential {
        f.write_all(tile_buf)?;
    } else {
        f.seek(SeekFrom::Start(entry.payload_offset))?;
        f.write_all(tile_buf)?;
    }
    Ok(())
}

fn encode_all_dataset_blobs(specs: &[ArrayWriteMeta<'_>]) -> Result<Vec<u8>, CatalogError> {
    let mut blob = Vec::new();
    for spec in specs {
        blob.extend_from_slice(&encode_dataset_blob(
            spec.name,
            spec.dtype,
            spec.shape,
            spec.chunk_shape,
        )?);
    }
    Ok(blob)
}

fn chunk_index_layout(
    dataset_blob_len: u64,
    n_chunks_total: u64,
) -> Result<(u64, u64, u64), CatalogError> {
    let index_base = wire::align8_u64(40u64 + dataset_blob_len);
    let index_header_len = index::CHUNK_INDEX_HEADER_V1.header_len as u64;
    let entries_len_u64 = n_chunks_total
        .checked_mul(ChunkIndexEntryV1::WIRE_LEN as u64)
        .ok_or(CatalogError::InvalidWriteSpec("chunk index size overflow"))?;
    let chunk_index_length = index_header_len
        .checked_add(entries_len_u64)
        .ok_or(CatalogError::InvalidWriteSpec("chunk index size overflow"))?;
    let payload_start = index_base + chunk_index_length;
    Ok((index_base, chunk_index_length, payload_start))
}

fn build_stream_index_and_jobs<'a>(
    specs: &'a [ArrayWriteMeta<'_>],
    payload_start: u64,
    n_chunks_total: u64,
) -> Result<(Vec<ChunkIndexEntryV1>, Vec<StreamTileJob<'a>>), CatalogError> {
    let mut entries: Vec<ChunkIndexEntryV1> =
        Vec::with_capacity(usize_from_u64("chunk_entry_count", n_chunks_total)?);
    let mut jobs: Vec<StreamTileJob<'_>> = Vec::with_capacity(entries.capacity());
    let mut cursor = payload_start;
    for (dataset_id, spec) in specs.iter().enumerate() {
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
                stored_byte_len: raw_len,
                codec: spec.chunk_codec,
            });
            jobs.push(StreamTileJob {
                dataset_id,
                dataset_name: spec.name,
                chunk_k: k,
                chunk_coord: chunk_index,
                ndim,
                raw_byte_len: raw_len,
            });
            cursor = cursor
                .checked_add(raw_len)
                .ok_or(CatalogError::InvalidWriteSpec("payload cursor overflow"))?;
        }
    }
    Ok((entries, jobs))
}

fn stream_superblock(
    specs: &[ArrayWriteMeta<'_>],
    index_base: u64,
    chunk_index_length: u64,
) -> Result<SuperblockV1, CatalogError> {
    Ok(SuperblockV1 {
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
    })
}

fn build_index_bytes(
    entries: &[ChunkIndexEntryV1],
    n_chunks_total: u64,
    chunk_index_length: u64,
    file_execution: FileExecutionSettingsV1,
) -> Result<Vec<u8>, CatalogError> {
    let mut index_bytes = Vec::with_capacity(usize_from_u64(
        "chunk_index_byte_length",
        chunk_index_length,
    )?);
    index::write_chunk_index_header(&mut index_bytes, n_chunks_total, file_execution);
    for e in entries {
        index_bytes.extend_from_slice(&e.to_bytes());
    }
    debug_assert_eq!(index_bytes.len() as u64, chunk_index_length);
    Ok(index_bytes)
}

fn write_file_preamble(
    f: &mut std::fs::File,
    sb: &SuperblockV1,
    blob: &[u8],
    index_base: u64,
    index_bytes: &[u8],
) -> Result<(), CatalogError> {
    f.write_all(&sb.to_bytes())?;
    f.write_all(&(blob.len() as u64).to_le_bytes())?;
    f.write_all(blob)?;
    let after_blob = 40usize + blob.len();
    let pad = usize_from_u64("chunk_index_base", index_base)?.saturating_sub(after_blob);
    if pad > 0 {
        f.write_all(&vec![0u8; pad])?;
    }
    f.write_all(index_bytes)?;
    Ok(())
}

fn entries_are_sequential(entries: &[ChunkIndexEntryV1]) -> bool {
    entries
        .windows(2)
        .all(|pair| pair[0].payload_offset + pair[0].raw_byte_len == pair[1].payload_offset)
}

fn prepare_tile_buffer(tile_buf: &mut Vec<u8>, raw_byte_len: u64) -> Result<(), CatalogError> {
    let len = usize_from_u64("tile_buffer", raw_byte_len)?;
    tile_buf.clear();
    tile_buf.try_reserve(len).map_err(std::io::Error::from)?;
    // SAFETY: `fill_tile` must write every byte before we flush the buffer.
    unsafe {
        tile_buf.set_len(len);
    }
    Ok(())
}

fn element_size_for_dtype(dtype: u32) -> Result<usize, CatalogError> {
    let tags = DATASET_DTYPE_TAG_V1;
    if tags.is_f32(dtype) {
        Ok(ElementDtype::F32.elem_size())
    } else if tags.is_f64(dtype) {
        Ok(ElementDtype::F64.elem_size())
    } else if tags.is_i32(dtype) {
        Ok(ElementDtype::I32.elem_size())
    } else if tags.is_i64(dtype) {
        Ok(ElementDtype::I64.elem_size())
    } else {
        Err(CatalogError::InvalidWriteSpec(
            "unsupported dtype for tile extraction",
        ))
    }
}
