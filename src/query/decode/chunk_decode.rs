//! Chunk payload I/O: mmap slices, codec decode, and per-element visitation.

use std::borrow::Cow;

use crate::catalog::{CHUNK_PAYLOAD_CODEC_V1, MAX_NDIM, tile};
use crate::query::decode::dense_visit::{dense_tile_logical_base, logical_index_unit_step};
use crate::query::decode::indexing::linear_rm_index;
use crate::query::fold::reduction::{ArgIndexAccum, ReductionKind, ValueAccum};
use crate::query::types::{PlannedChunkIo, ReadPlan, TetError};
use crate::utils::{f32_le, f64_le, i32_le, i64_le, wire};

struct PreparedChunk<'a> {
    bytes_read: u64,
    raw_bytes: Cow<'a, [u8]>,
    nelem: usize,
    tile: Vec<u64>,
    chunk_coord: [u64; MAX_NDIM],
    ndim: usize,
}

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

fn prepare_chunk<'a>(
    mmap: &'a [u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    elem_size: usize,
) -> Result<PreparedChunk<'a>, TetError> {
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
    let nelem = raw_bytes.len() / elem_size;
    let mut chunk_coord = [0u64; MAX_NDIM];
    chunk_coord[..ndim].copy_from_slice(&c.chunk_index[..ndim]);
    let tile = tile::tile_extent(&plan.dataset_shape, &plan.chunk_shape, &chunk_coord, ndim);
    let tile_elems: u64 = tile
        .iter()
        .try_fold(1u64, |a, &b| a.checked_mul(b))
        .ok_or_else(|| TetError::Validation("tile element count overflow".into()))?;
    let tile_elems_us = usize::try_from(tile_elems)
        .map_err(|_| TetError::Validation("tile element count too large for this host".into()))?;
    if tile_elems_us != nelem {
        return Err(TetError::Validation(format!(
            "chunk_index={:?}: tile has {tile_elems_us} values but raw_byte_len implies {nelem}",
            c.chunk_index
        )));
    }
    Ok(PreparedChunk {
        bytes_read,
        raw_bytes,
        nelem,
        tile,
        chunk_coord,
        ndim,
    })
}

fn logical_index_for_global(
    plan: &ReadPlan,
    global: &[u64],
    ndim: usize,
) -> Result<usize, TetError> {
    if plan.selection_step.iter().all(|&s| s == 1) {
        return logical_index_unit_step(plan, global, ndim);
    }
    let mut lc = [0usize; MAX_NDIM];
    for (d, &x) in global.iter().enumerate().take(ndim) {
        let st = plan.selection_box_start[d];
        let q = (x - st) / plan.selection_step[d];
        lc[d] = usize::try_from(q).map_err(|_| {
            TetError::Validation(format!(
                "logical coordinate on axis {d} does not fit usize on this host"
            ))
        })?;
    }
    linear_rm_index(&lc[..ndim], &plan.logical_selection_shape)
}

fn visit_prepared_chunk_dense<F>(
    prep: &PreparedChunk<'_>,
    li_base: usize,
    read_f32: fn(&[u8], usize) -> f32,
    mut visit: F,
) -> Result<(), TetError>
where
    F: FnMut(usize, f32) -> Result<(), TetError>,
{
    for k in 0..prep.nelem {
        visit(li_base + k, read_f32(&prep.raw_bytes, k))?;
    }
    Ok(())
}

fn visit_prepared_chunk_strided_f32<F>(
    prep: &PreparedChunk<'_>,
    plan: &ReadPlan,
    read_f32: fn(&[u8], usize) -> f32,
    mut visit: F,
) -> Result<(), TetError>
where
    F: FnMut(usize, f32) -> Result<(), TetError>,
{
    let ndim = prep.ndim;
    let mut global = [0u64; MAX_NDIM];
    for k in 0..prep.nelem {
        let local = tile::local_coords_from_linear(k as u64, &prep.tile, ndim);
        for (d, gv) in global.iter_mut().enumerate().take(ndim) {
            *gv = prep.chunk_coord[d]
                .saturating_mul(plan.chunk_shape[d])
                .saturating_add(local[d]);
        }
        if !global_matches_strided_selection(&global[..ndim], plan) {
            continue;
        }
        let li = logical_index_for_global(plan, &global[..ndim], ndim)?;
        visit(li, read_f32(&prep.raw_bytes, k))?;
    }
    Ok(())
}

fn visit_prepared_chunk_strided_typed<T, F>(
    prep: &PreparedChunk<'_>,
    plan: &ReadPlan,
    read_at: fn(&[u8], usize) -> T,
    mut visit: F,
) -> Result<(), TetError>
where
    F: FnMut(usize, T) -> Result<(), TetError>,
{
    let ndim = prep.ndim;
    let mut global = [0u64; MAX_NDIM];
    for k in 0..prep.nelem {
        let local = tile::local_coords_from_linear(k as u64, &prep.tile, ndim);
        for (d, gv) in global.iter_mut().enumerate().take(ndim) {
            *gv = prep.chunk_coord[d]
                .saturating_mul(plan.chunk_shape[d])
                .saturating_add(local[d]);
        }
        if !global_matches_strided_selection(&global[..ndim], plan) {
            continue;
        }
        let li = logical_index_for_global(plan, &global[..ndim], ndim)?;
        visit(li, read_at(&prep.raw_bytes, k))?;
    }
    Ok(())
}

fn visit_prepared_chunk_f32<F>(
    prep: &PreparedChunk<'_>,
    plan: &ReadPlan,
    visit: F,
) -> Result<(), TetError>
where
    F: FnMut(usize, f32) -> Result<(), TetError>,
{
    if prep.ndim == 1
        && let Some(li_base) =
            dense_tile_logical_base(plan, &prep.chunk_coord[..prep.ndim], &prep.tile)?
    {
        return visit_prepared_chunk_dense(prep, li_base, f32_le::read_f32_le_at, visit);
    }
    visit_prepared_chunk_strided_f32(prep, plan, f32_le::read_f32_le_at, visit)
}

fn visit_prepared_chunk_typed<T, F>(
    prep: &PreparedChunk<'_>,
    plan: &ReadPlan,
    read_at: fn(&[u8], usize) -> T,
    mut visit: F,
) -> Result<(), TetError>
where
    F: FnMut(usize, T) -> Result<(), TetError>,
{
    if prep.ndim == 1
        && let Some(li_base) =
            dense_tile_logical_base(plan, &prep.chunk_coord[..prep.ndim], &prep.tile)?
    {
        for k in 0..prep.nelem {
            visit(li_base + k, read_at(&prep.raw_bytes, k))?;
        }
        return Ok(());
    }
    visit_prepared_chunk_strided_typed(prep, plan, read_at, visit)
}

/// Scalar fold over one planned `f32` chunk without per-element visitor callbacks.
pub(crate) fn fold_planned_chunk_f32(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    kind: ReductionKind,
    value: &mut ValueAccum,
    arg: &mut ArgIndexAccum,
    preview: &mut [f32],
) -> Result<u64, TetError> {
    let prep = prepare_chunk(mmap, plan, c, 4)?;
    let preview_len = preview.len();
    if prep.ndim == 1
        && let Some(li_base) =
            dense_tile_logical_base(plan, &prep.chunk_coord[..prep.ndim], &prep.tile)?
    {
        match kind {
            ReductionKind::ArgMin | ReductionKind::ArgMax => {
                for k in 0..prep.nelem {
                    let v = f32_le::read_f32_le_at(&prep.raw_bytes, k);
                    let li = li_base + k;
                    arg.push(li as u64, v, kind);
                    if li < preview_len {
                        preview[li] = v;
                    }
                }
            }
            _ => {
                value.push_f32_le_bytes(&prep.raw_bytes, kind);
                if preview_len > 0 {
                    let cap = preview_len.saturating_sub(li_base).min(prep.nelem);
                    for k in 0..cap {
                        preview[li_base + k] = f32_le::read_f32_le_at(&prep.raw_bytes, k);
                    }
                }
            }
        }
        return Ok(prep.bytes_read);
    }
    visit_prepared_chunk_f32(&prep, plan, |li, v| {
        match kind {
            ReductionKind::ArgMin | ReductionKind::ArgMax => arg.push(li as u64, v, kind),
            _ => value.push(v),
        }
        if li < preview_len {
            preview[li] = v;
        }
        Ok(())
    })?;
    Ok(prep.bytes_read)
}

/// Scalar fold over one planned `f64` chunk without per-element visitor callbacks.
pub(crate) fn fold_planned_chunk_f64(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    kind: ReductionKind,
    value: &mut ValueAccum,
    arg: &mut ArgIndexAccum,
    preview: &mut [f64],
) -> Result<u64, TetError> {
    let prep = prepare_chunk(mmap, plan, c, 8)?;
    let preview_len = preview.len();
    if prep.ndim == 1
        && let Some(li_base) =
            dense_tile_logical_base(plan, &prep.chunk_coord[..prep.ndim], &prep.tile)?
    {
        match kind {
            ReductionKind::ArgMin | ReductionKind::ArgMax => {
                for k in 0..prep.nelem {
                    let v = f64_le::read_f64_le_at(&prep.raw_bytes, k);
                    let li = li_base + k;
                    arg.push_f64(li as u64, v, kind);
                    if li < preview_len {
                        preview[li] = v;
                    }
                }
            }
            _ => {
                value.push_f64_le_bytes(&prep.raw_bytes, kind);
                if preview_len > 0 {
                    let cap = preview_len.saturating_sub(li_base).min(prep.nelem);
                    for k in 0..cap {
                        preview[li_base + k] = f64_le::read_f64_le_at(&prep.raw_bytes, k);
                    }
                }
            }
        }
        return Ok(prep.bytes_read);
    }
    visit_prepared_chunk_typed(&prep, plan, f64_le::read_f64_le_at, |li, v| {
        match kind {
            ReductionKind::ArgMin | ReductionKind::ArgMax => arg.push_f64(li as u64, v, kind),
            _ => value.push_f64(v),
        }
        if li < preview_len {
            preview[li] = v;
        }
        Ok(())
    })?;
    Ok(prep.bytes_read)
}

/// Visit each selected `f32` in a planned chunk.
pub(crate) fn visit_planned_chunk<F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, f32) -> Result<(), TetError>,
{
    let prep = prepare_chunk(mmap, plan, c, 4)?;
    visit_prepared_chunk_f32(&prep, plan, visit)?;
    Ok(prep.bytes_read)
}

fn visit_planned_chunk_typed<T, F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    elem_size: usize,
    read_at: fn(&[u8], usize) -> T,
    visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, T) -> Result<(), TetError>,
{
    let prep = prepare_chunk(mmap, plan, c, elem_size)?;
    visit_prepared_chunk_typed(&prep, plan, read_at, visit)?;
    Ok(prep.bytes_read)
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

/// Visit each selected `i32`, promoting values to `f64` for accumulators.
pub(crate) fn visit_planned_chunk_i32_as_f64<F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    mut visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, f64) -> Result<(), TetError>,
{
    visit_planned_chunk_i32(mmap, plan, c, |li, v| visit(li, f64::from(v)))
}

/// Visit each selected `i64`, promoting values to `f64` for accumulators.
pub(crate) fn visit_planned_chunk_i64_as_f64<F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    mut visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, f64) -> Result<(), TetError>,
{
    visit_planned_chunk_i64(mmap, plan, c, |li, v| visit(li, v as f64))
}

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

pub(crate) fn visit_planned_chunk_i32<F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, i32) -> Result<(), TetError>,
{
    visit_planned_chunk_typed(mmap, plan, c, 4, i32_le::read_i32_le_at, visit)
}

pub(crate) fn visit_planned_chunk_i64<F>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, i64) -> Result<(), TetError>,
{
    visit_planned_chunk_typed(mmap, plan, c, 8, i64_le::read_i64_le_at, visit)
}

pub(crate) fn scatter_chunk_into_plan_i32(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    out: &mut [Option<i32>],
) -> Result<u64, TetError> {
    visit_planned_chunk_i32(mmap, plan, c, |li, v| {
        if li < out.len() {
            out[li] = Some(v);
        }
        Ok(())
    })
}

pub(crate) fn scatter_chunk_into_plan_i64(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    out: &mut [Option<i64>],
) -> Result<u64, TetError> {
    visit_planned_chunk_i64(mmap, plan, c, |li, v| {
        if li < out.len() {
            out[li] = Some(v);
        }
        Ok(())
    })
}
