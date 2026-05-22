//! Decode planned chunk payloads into logical row-major `f32`.

use crate::query::types::{ReadPlan, TetError};

use super::chunk_decode::{scatter_chunk_into_plan, visit_planned_chunk};
use super::fold::{build_fold_plan_outcome, validate_fold_preview};
use super::reduction::{ReductionKind, ValueAccum};

pub(crate) use super::fold::FoldPlanOutcome;

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
) -> Result<FoldPlanOutcome, TetError> {
    let n = plan.logical_f32_element_count;
    let preview_cap = max_f32.min(n);
    let mut preview = vec![f32::NAN; preview_cap];
    let mut acc = ValueAccum::default();
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

    validate_fold_preview(!acc.is_empty(), &preview, preview_cap)?;

    Ok(build_fold_plan_outcome(
        preview,
        max_f32,
        n,
        total_bytes_read_from_disk,
        acc.finish_scalar(kind).into(),
    ))
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

/// Spill the full logical selection as row-major `f32` LE to `path` using a file-backed mmap
/// (disk-resident; does not allocate a dense `Vec` in RAM).
///
/// # Errors
///
/// Same validation failures as [`materialize_read_plan_f32_le`], plus I/O or mmap errors on `path`.
pub fn spill_read_plan_f32_le(
    mmap: &[u8],
    plan: &ReadPlan,
    path: &std::path::Path,
) -> Result<u64, TetError> {
    use std::fs::OpenOptions;
    use std::io::Write;

    use memmap2::MmapMut;

    let n = plan.logical_f32_element_count;
    let byte_len = u64::try_from(n)
        .map_err(|_| TetError::Validation("logical element count overflow".into()))?;
    let byte_len = crate::utils::f32_le::bytes_from_elem_count(byte_len)
        .ok_or_else(|| TetError::Validation("spill byte length overflow".into()))?;
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .map_err(|e| TetError::Validation(format!("spill open failed: {e}")))?;
    file.set_len(byte_len)
        .map_err(|e| TetError::Validation(format!("spill set_len failed: {e}")))?;
    file.flush()
        .map_err(|e| TetError::Validation(format!("spill flush failed: {e}")))?;
    let mut out_mmap = unsafe {
        MmapMut::map_mut(&file)
            .map_err(|e| TetError::Validation(format!("spill mmap failed: {e}")))?
    };
    let out = bytemuck::cast_slice_mut(out_mmap.as_mut());
    validate_read_plan_geometry(plan, out.len())?;
    let total = materialize_scatter_fill(mmap, plan, out)?;
    check_materialized_complete(out)?;
    out_mmap
        .flush()
        .map_err(|e| TetError::Validation(format!("spill mmap flush failed: {e}")))?;
    Ok(total)
}
