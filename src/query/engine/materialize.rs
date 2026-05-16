//! Decode planned chunk payloads into logical row-major `f32`.

use std::borrow::Cow;

use crate::catalog::{CHUNK_PAYLOAD_CODEC_V1, MAX_NDIM, tile};
use crate::query::types::{PlannedChunkIo, ReadPlan, TetError};

use super::indexing::linear_rm_index;
use super::reduction::{ReductionKind, ScalarAccum, ScalarReductionResult};

fn u64_to_usize(field: &'static str, v: u64) -> Result<usize, TetError> {
    usize::try_from(v)
        .map_err(|_| TetError::Validation(format!("{field}={v} is too large for this platform")))
}

pub(crate) fn planned_chunk_stored_slice<'a>(
    mmap: &'a [u8],
    c: &PlannedChunkIo,
) -> Result<&'a [u8], TetError> {
    let off = u64_to_usize("payload_offset", c.payload_offset)?;
    let len = u64_to_usize("stored_byte_len", c.stored_byte_len)?;
    let end = off
        .checked_add(len)
        .ok_or_else(|| TetError::Validation("payload byte range overflow".into()))?;
    if end > mmap.len() {
        return Err(TetError::Validation(format!(
            "chunk_index={:?}: payload byte range [{off},{end}) extends past mmap length {}",
            c.chunk_index,
            mmap.len()
        )));
    }
    Ok(&mmap[off..end])
}

pub(crate) fn decode_planned_chunk_bytes<'a>(
    stored: &'a [u8],
    c: &PlannedChunkIo,
) -> Result<Cow<'a, [u8]>, TetError> {
    if CHUNK_PAYLOAD_CODEC_V1.is_raw(c.codec) {
        if c.stored_byte_len != c.raw_byte_len {
            return Err(TetError::Validation(format!(
                "raw codec requires stored_byte_len == raw_byte_len for chunk_index={:?}",
                c.chunk_index
            )));
        }
        return Ok(Cow::Borrowed(stored));
    }
    if CHUNK_PAYLOAD_CODEC_V1.is_zstd(c.codec) {
        let dec = zstd::decode_all(stored).map_err(|e| {
            TetError::Validation(format!(
                "zstd decode failed for chunk_index={:?}: {e}",
                c.chunk_index
            ))
        })?;
        if dec.len() as u64 != c.raw_byte_len {
            return Err(TetError::Validation(format!(
                "zstd decoded length {} != raw_byte_len {} for chunk_index={:?}",
                dec.len(),
                c.raw_byte_len,
                c.chunk_index
            )));
        }
        return Ok(Cow::Owned(dec));
    }
    Err(TetError::Validation(format!(
        "unsupported codec {} for chunk_index={:?} (supported: {} raw, {} zstd)",
        c.codec, c.chunk_index, CHUNK_PAYLOAD_CODEC_V1.raw, CHUNK_PAYLOAD_CODEC_V1.zstd
    )))
}

/// Map each planned chunk to a subslice of `mmap` (zero-copy).
///
/// # Errors
///
/// Returns [`TetError::Validation`] when a chunk is not mmap-readable as raw bytes (`codec` must be
/// [`CHUNK_PAYLOAD_CODEC_V1.raw`](crate::catalog::ChunkPayloadCodecV1::raw)), lengths disagree, ranges overflow, or payload bytes fall outside `mmap`.
///
/// For [`CHUNK_PAYLOAD_CODEC_V1.zstd`](crate::catalog::ChunkPayloadCodecV1::zstd) payloads use [`materialize_read_plan_f32_le`] (or another decode path);
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

type ScatterFillFn = fn(&[u8], &ReadPlan, &mut [f32]) -> Result<u64, TetError>;

pub(crate) fn materialize_read_plan_f32_le_core(
    mmap: &[u8],
    plan: &ReadPlan,
    max_elements: Option<usize>,
    scatter_fill: ScatterFillFn,
) -> Result<(Vec<f32>, bool, u64), TetError> {
    if matches!(max_elements, Some(0)) {
        return Ok((Vec::new(), false, 0));
    }
    let n = plan.logical_f32_element_count;
    let mut out = vec![f32::NAN; n];
    let total_bytes_read_from_disk = scatter_fill(mmap, plan, &mut out)?;
    check_materialized_complete(&out)?;
    let truncated = max_elements.is_some_and(|cap| cap < n);
    if let Some(cap) = max_elements {
        out.truncate(cap.min(n));
    }
    Ok((out, truncated, total_bytes_read_from_disk))
}

pub(crate) fn materialize_read_plan_f32_le_into_core(
    mmap: &[u8],
    plan: &ReadPlan,
    max_elements: Option<usize>,
    dst: &mut [f32],
    scatter_fill: ScatterFillFn,
) -> Result<MaterializeReadPlanF32IntoOutcome, TetError> {
    let n = plan.logical_f32_element_count;
    if matches!(max_elements, Some(0)) {
        return Ok(MaterializeReadPlanF32IntoOutcome {
            logical_element_count: n,
            elements_written: 0,
            truncated: n > 0,
            total_bytes_read_from_disk: 0,
        });
    }
    let want_write = max_elements.map_or(n, |m| m.min(n));
    if dst.len() < want_write {
        return Err(TetError::Validation(format!(
            "destination buffer length {} < required {} (logical element count {})",
            dst.len(),
            want_write,
            n
        )));
    }
    let mut tmp = vec![f32::NAN; n];
    let total_bytes_read_from_disk = scatter_fill(mmap, plan, &mut tmp)?;
    check_materialized_complete(&tmp)?;
    dst[..want_write].copy_from_slice(&tmp[..want_write]);
    Ok(MaterializeReadPlanF32IntoOutcome {
        logical_element_count: n,
        elements_written: want_write,
        truncated: max_elements.is_some_and(|m| m < n),
        total_bytes_read_from_disk,
    })
}

fn check_materialized_complete(out: &[f32]) -> Result<(), TetError> {
    if out.iter().any(|v| v.is_nan()) {
        return Err(TetError::Validation(
            "materialized selection has unset elements (chunk payloads vs selection mismatch)"
                .into(),
        ));
    }
    Ok(())
}

/// Decode planned raw `f32` chunk payloads (little-endian) into **logical row-major** order for the
/// strided selection encoded on [`ReadPlan`].
///
/// `max_elements`: `None` decodes every float in the logical tensor. `Some(0)` returns an empty
/// vector and reads nothing from disk. `Some(n)` for `n > 0` returns the first `n` values in
/// logical row-major order and sets `truncated` when the logical tensor is longer.
///
/// # Errors
///
/// Returns [`TetError::Validation`] when chunk payloads disagree with tile geometry, the
/// strided selection is not fully covered by planned chunks, mmap bounds fail, or zstd decode
/// fails.
pub fn materialize_read_plan_f32_le(
    mmap: &[u8],
    plan: &ReadPlan,
    max_elements: Option<usize>,
) -> Result<(Vec<f32>, bool, u64), TetError> {
    materialize_read_plan_f32_le_core(mmap, plan, max_elements, materialize_scatter_fill)
}

/// Outcome of [`materialize_read_plan_f32_le_into`].
#[derive(Debug, Clone)]
pub struct MaterializeReadPlanF32IntoOutcome {
    /// Logical tensor element count (selection grid product).
    pub logical_element_count: usize,
    /// Values written to the start of the caller buffer (`min(max_elements.unwrap_or(logical), logical)`).
    pub elements_written: usize,
    pub truncated: bool,
    pub total_bytes_read_from_disk: u64,
}

/// Like [`materialize_read_plan_f32_le`], but writes decoded values into `dst` without allocating a `Vec`.
///
/// When `max_elements` is `None`, `dst.len()` must be at least [`ReadPlan::logical_f32_element_count`].
/// When `max_elements` is `Some(m)` with `m > 0`, `dst.len()` must be at least `m.min(logical)`.
/// `Some(0)` writes nothing and does not touch `dst`.
///
/// # Errors
///
/// Same failure modes as [`materialize_read_plan_f32_le`], plus a short destination buffer.
pub fn materialize_read_plan_f32_le_into(
    mmap: &[u8],
    plan: &ReadPlan,
    max_elements: Option<usize>,
    dst: &mut [f32],
) -> Result<MaterializeReadPlanF32IntoOutcome, TetError> {
    materialize_read_plan_f32_le_into_core(mmap, plan, max_elements, dst, materialize_scatter_fill)
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

pub(crate) fn validate_read_plan_geometry(plan: &ReadPlan, out_len: usize) -> Result<(), TetError> {
    let ndim = plan.dataset_shape.len();
    if plan.chunk_shape.len() != ndim
        || plan.selection_box_start.len() != ndim
        || plan.selection_box_stop_exclusive.len() != ndim
        || plan.selection_step.len() != ndim
        || plan.logical_selection_shape.len() != ndim
    {
        return Err(TetError::Validation(
            "read_plan geometry fields have inconsistent rank".into(),
        ));
    }
    if out_len != plan.logical_f32_element_count {
        return Err(TetError::Validation(format!(
            "output buffer length {out_len} != read_plan.logical_f32_element_count {}",
            plan.logical_f32_element_count
        )));
    }
    Ok(())
}

/// Result of [`fold_read_plan_scalar_operation`].
#[derive(Debug, Clone)]
pub(crate) struct FoldScalarPlanOutcome {
    pub f32_preview: Vec<f32>,
    pub f32_preview_truncated: bool,
    pub total_bytes_read_from_disk: u64,
    pub scalar: ScalarReductionResult,
}

/// Decode planned chunks once, aggregating a scalar reduction without allocating the full
/// logical tensor. Fills `f32_preview` with the first `max_f32` logical row-major values.
///
/// # Errors
///
/// Same validation failures as materialization when chunk payloads disagree with the plan.
pub(crate) fn fold_read_plan_scalar_operation(
    mmap: &[u8],
    plan: &ReadPlan,
    max_f32: usize,
    kind: ReductionKind,
) -> Result<FoldScalarPlanOutcome, TetError> {
    let n = plan.logical_f32_element_count;
    let preview_cap = max_f32.min(n);
    let mut preview = vec![f32::NAN; preview_cap];
    let mut acc = ScalarAccum::default();
    let mut total_bytes_read_from_disk: u64 = 0;

    for c in &plan.chunks {
        let chunk_bytes = visit_planned_chunk(mmap, plan, c, |li, v| {
            acc.push(v);
            if li < preview_cap {
                preview[li] = v;
            }
            Ok(())
        })?;
        total_bytes_read_from_disk = total_bytes_read_from_disk
            .checked_add(chunk_bytes)
            .ok_or_else(|| TetError::Validation("total bytes read overflow".into()))?;
    }

    if acc.is_empty() {
        return Err(TetError::Validation(
            "operation requires at least one decoded f32 from the read plan".into(),
        ));
    }
    if preview_cap > 0 && preview.iter().any(|v| v.is_nan()) {
        return Err(TetError::Validation(
            "materialized selection has unset preview elements (chunk payloads vs selection mismatch)"
                .into(),
        ));
    }

    Ok(FoldScalarPlanOutcome {
        f32_preview: if max_f32 == 0 { Vec::new() } else { preview },
        f32_preview_truncated: n > max_f32,
        total_bytes_read_from_disk,
        scalar: acc.finish(kind),
    })
}

/// Visit each selected `f32` in a planned chunk.
///
/// Returns stored payload bytes read from `mmap` for this chunk.
fn visit_planned_chunk<F>(
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
        let a: [u8; 4] = raw_bytes[k * 4..k * 4 + 4]
            .try_into()
            .map_err(|_| TetError::Validation("internal: f32 slice chunking".into()))?;
        visit(li, f32::from_le_bytes(a))?;
    }
    Ok(bytes_read)
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
        out[li] = v;
        Ok(())
    })
}

pub(crate) fn materialize_scatter_fill(
    mmap: &[u8],
    plan: &ReadPlan,
    out: &mut [f32],
) -> Result<u64, TetError> {
    validate_read_plan_geometry(plan, out.len())?;
    let mut total_bytes_read_from_disk: u64 = 0;
    for c in &plan.chunks {
        let n = scatter_chunk_into_plan(mmap, plan, c, out)?;
        total_bytes_read_from_disk = total_bytes_read_from_disk
            .checked_add(n)
            .ok_or_else(|| TetError::Validation("total bytes read overflow".into()))?;
    }
    Ok(total_bytes_read_from_disk)
}
