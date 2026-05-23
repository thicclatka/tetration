//! Chunk payload I/O: mmap slices, codec decode, and per-element visitation.

use std::borrow::Cow;

use crate::catalog::{CHUNK_PAYLOAD_CODEC_V1, MAX_NDIM, tile};
use crate::query::types::{PlannedChunkIo, ReadPlan, TetError};
use crate::utils::{f32_le, f64_le, wire};

use super::indexing::linear_rm_index;

fn u64_to_usize(field: &'static str, v: u64) -> Result<usize, TetError> {
    usize::try_from(v)
        .map_err(|_| TetError::Validation(format!("{field}={v} is too large for this platform")))
}

fn map_span_err(e: wire::SpanError, chunk_index: &[u64], mmap_len: usize) -> TetError {
    match e {
        wire::SpanError::AddOverflow => TetError::Validation("payload byte range overflow".into()),
        wire::SpanError::OutOfBounds { end } => TetError::Validation(format!(
            "chunk_index={chunk_index:?}: payload byte range extends past mmap length {mmap_len} (end={end})"
        )),
    }
}

fn map_codec_err(e: &crate::catalog::CatalogError, chunk_index: &[u64]) -> TetError {
    TetError::Validation(format!("chunk_index={chunk_index:?}: {e}"))
}

pub(crate) fn planned_chunk_stored_slice<'a>(
    mmap: &'a [u8],
    c: &PlannedChunkIo,
) -> Result<&'a [u8], TetError> {
    let off = u64_to_usize("payload_offset", c.payload_offset)?;
    let len = u64_to_usize("stored_byte_len", c.stored_byte_len)?;
    let range = wire::checked_usize_subslice(off, len, mmap.len())
        .map_err(|e| map_span_err(e, &c.chunk_index, mmap.len()))?;
    Ok(&mmap[range])
}

pub(crate) fn decode_planned_chunk_bytes<'a>(
    stored: &'a [u8],
    c: &PlannedChunkIo,
) -> Result<Cow<'a, [u8]>, TetError> {
    CHUNK_PAYLOAD_CODEC_V1
        .decode_tile_payload(stored, c.raw_byte_len, c.stored_byte_len, c.codec)
        .map_err(|e| map_codec_err(&e, &c.chunk_index))
}

/// Map each planned chunk to a subslice of `mmap` (zero-copy).
///
/// # Errors
///
/// Returns [`TetError::Validation`] when a chunk is not mmap-readable as raw bytes (`codec` must be
/// [`CHUNK_PAYLOAD_CODEC_V1.raw`](crate::catalog::ChunkPayloadCodecV1::raw)), lengths disagree, ranges overflow, or payload bytes fall outside `mmap`.
///
/// For [`CHUNK_PAYLOAD_CODEC_V1.zstd`](crate::catalog::ChunkPayloadCodecV1::zstd) payloads use [`super::materialize::materialize_read_plan_f32_le`] (or another decode path);
/// compressed bytes are not returned as a single mmap slice here.
pub fn planned_chunk_mmap_slices<'a>(
    mmap: &'a [u8],
    plan: &ReadPlan,
) -> Result<Vec<&'a [u8]>, TetError> {
    let mut out = Vec::with_capacity(plan.chunks.len());
    for c in &plan.chunks {
        if !CHUNK_PAYLOAD_CODEC_V1.is_raw(c.codec) {
            return Err(TetError::Validation(format!(
                "planned_chunk_mmap_slices requires codec {} (raw); got codec={} for chunk_index={:?}",
                CHUNK_PAYLOAD_CODEC_V1.raw, c.codec, c.chunk_index
            )));
        }
        let stored = planned_chunk_stored_slice(mmap, c)?;
        decode_planned_chunk_bytes(stored, c)?;
        out.push(stored);
    }
    Ok(out)
}

fn global_matches_strided_selection(g: &[u64], plan: &ReadPlan) -> bool {
    (0..g.len()).all(|d| {
        let x = g[d];
        let st = plan.selection_box_start[d];
        let hi = plan.selection_box_stop_exclusive[d];
        let step = plan.selection_step[d];
        x >= st && x < hi && (x - st).is_multiple_of(step)
    })
}

/// Visit each selected `f32` in a planned chunk.
///
/// Returns stored payload bytes read from `mmap` for this chunk.
pub(crate) fn visit_planned_chunk<F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    mut visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, f32) -> Result<(), TetError>,
{
    let ndim = plan.dataset_shape.len();
    if c.chunk_index.len() != ndim {
        return Err(TetError::Validation(format!(
            "chunk_index={:?} rank {} != dataset rank {}",
            c.chunk_index,
            c.chunk_index.len(),
            ndim
        )));
    }
    let stored = planned_chunk_stored_slice(mmap, c)?;
    let bytes_read = stored.len() as u64;
    let raw_bytes = decode_planned_chunk_bytes(stored, c)?;
    if !raw_bytes.len().is_multiple_of(4) {
        return Err(TetError::Validation(format!(
            "chunk raw length {} is not a multiple of 4 for f32 (chunk_index={:?})",
            raw_bytes.len(),
            c.chunk_index
        )));
    }
    let nelem_tile = raw_bytes.len() / 4;
    let mut coord = [0u64; MAX_NDIM];
    coord[..ndim].copy_from_slice(&c.chunk_index[..ndim]);
    let tile = tile::tile_extent(&plan.dataset_shape, &plan.chunk_shape, &coord, ndim);
    let tile_elems: u64 = tile
        .iter()
        .try_fold(1u64, |a, &b| a.checked_mul(b))
        .ok_or_else(|| TetError::Validation("tile element count overflow".into()))?;
    let tile_elems_us = usize::try_from(tile_elems)
        .map_err(|_| TetError::Validation("tile element count too large for this host".into()))?;
    if tile_elems_us != nelem_tile {
        return Err(TetError::Validation(format!(
            "chunk_index={:?}: tile has {tile_elems_us} f32 values but raw_byte_len implies {nelem_tile}",
            c.chunk_index
        )));
    }
    for k in 0..tile_elems_us {
        let local = tile::local_coords_from_linear(k as u64, &tile, ndim);
        let mut global = [0u64; MAX_NDIM];
        for (d, gv) in global.iter_mut().enumerate().take(ndim) {
            *gv = coord[d]
                .saturating_mul(plan.chunk_shape[d])
                .saturating_add(local[d]);
        }
        if !global_matches_strided_selection(&global[..ndim], plan) {
            continue;
        }
        let mut lc = Vec::with_capacity(ndim);
        for (d, &x) in global[..ndim].iter().enumerate() {
            let st = plan.selection_box_start[d];
            let q = (x - st) / plan.selection_step[d];
            let qi = usize::try_from(q).map_err(|_| {
                TetError::Validation(format!(
                    "logical coordinate on axis {d} does not fit usize on this host"
                ))
            })?;
            lc.push(qi);
        }
        let li = linear_rm_index(&lc, &plan.logical_selection_shape)?;
        visit(li, f32_le::read_f32_le_at(&raw_bytes, k))?;
    }
    Ok(bytes_read)
}

fn visit_planned_chunk_typed<F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    elem_size: usize,
    read_at: fn(&[u8], usize) -> f64,
    mut visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, f64) -> Result<(), TetError>,
{
    let ndim = plan.dataset_shape.len();
    if c.chunk_index.len() != ndim {
        return Err(TetError::Validation(format!(
            "chunk_index={:?} rank {} != dataset rank {}",
            c.chunk_index,
            c.chunk_index.len(),
            ndim
        )));
    }
    let stored = planned_chunk_stored_slice(mmap, c)?;
    let bytes_read = stored.len() as u64;
    let raw_bytes = decode_planned_chunk_bytes(stored, c)?;
    if !raw_bytes.len().is_multiple_of(elem_size) {
        return Err(TetError::Validation(format!(
            "chunk raw length {} is not a multiple of {elem_size} (chunk_index={:?})",
            raw_bytes.len(),
            c.chunk_index
        )));
    }
    let nelem_tile = raw_bytes.len() / elem_size;
    let mut coord = [0u64; MAX_NDIM];
    coord[..ndim].copy_from_slice(&c.chunk_index[..ndim]);
    let tile = tile::tile_extent(&plan.dataset_shape, &plan.chunk_shape, &coord, ndim);
    let tile_elems: u64 = tile
        .iter()
        .try_fold(1u64, |a, &b| a.checked_mul(b))
        .ok_or_else(|| TetError::Validation("tile element count overflow".into()))?;
    let tile_elems_us = usize::try_from(tile_elems)
        .map_err(|_| TetError::Validation("tile element count too large for this host".into()))?;
    if tile_elems_us != nelem_tile {
        return Err(TetError::Validation(format!(
            "chunk_index={:?}: tile has {tile_elems_us} values but raw_byte_len implies {nelem_tile}",
            c.chunk_index
        )));
    }
    for k in 0..tile_elems_us {
        let local = tile::local_coords_from_linear(k as u64, &tile, ndim);
        let mut global = [0u64; MAX_NDIM];
        for (d, gv) in global.iter_mut().enumerate().take(ndim) {
            *gv = coord[d]
                .saturating_mul(plan.chunk_shape[d])
                .saturating_add(local[d]);
        }
        if !global_matches_strided_selection(&global[..ndim], plan) {
            continue;
        }
        let mut lc = Vec::with_capacity(ndim);
        for (d, &x) in global[..ndim].iter().enumerate() {
            let st = plan.selection_box_start[d];
            let q = (x - st) / plan.selection_step[d];
            let qi = usize::try_from(q).map_err(|_| {
                TetError::Validation(format!(
                    "logical coordinate on axis {d} does not fit usize on this host"
                ))
            })?;
            lc.push(qi);
        }
        let li = linear_rm_index(&lc, &plan.logical_selection_shape)?;
        visit(li, read_at(&raw_bytes, k))?;
    }
    Ok(bytes_read)
}

/// Visit each selected `f64` in a planned chunk.
pub(crate) fn visit_planned_chunk_f64<F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, f64) -> Result<(), TetError>,
{
    visit_planned_chunk_typed(mmap, plan, c, 8, f64_le::read_f64_le_at, visit)
}

/// Decode one planned chunk and scatter matching `f64` values into `out`.
pub(crate) fn scatter_chunk_into_plan_f64(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    out: &mut [f64],
) -> Result<u64, TetError> {
    visit_planned_chunk_f64(mmap, plan, c, |li, v| {
        if li < out.len() {
            out[li] = v;
        }
        Ok(())
    })
}

/// Decode one planned chunk and scatter matching `f32` values into `out`.
///
/// Returns stored payload bytes read from `mmap` for this chunk.
pub(crate) fn scatter_chunk_into_plan(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    out: &mut [f32],
) -> Result<u64, TetError> {
    visit_planned_chunk(mmap, plan, c, |li, v| {
        if li < out.len() {
            out[li] = v;
        }
        Ok(())
    })
}
