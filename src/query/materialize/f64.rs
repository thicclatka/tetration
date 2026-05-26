//! `f64` logical materialize, spill, and scalar fold.

use crate::query::decode::chunk_decode::{scatter_chunk_into_plan_f64, visit_planned_chunk_f64};
use crate::query::fold::FoldPlanOutcome;
use crate::query::fold::reduction::{ArgIndexAccum, ReductionKind, ValueAccum};
use crate::query::fold::shared::{FoldPreviewBuffer, build_fold_plan_outcome_typed};
use crate::query::types::{ReadPlan, TetError};

use super::parallel::materialize_read_plan_f64_le_parallel;
use crate::utils::f64_le;

use super::validate::{validate_full_read_plan_buffer, validate_read_plan_geometry};

// --- f64 materialize / fold / spill ---

type ScatterFillF64Fn = fn(&[u8], &ReadPlan, &mut [f64]) -> Result<u64, TetError>;

fn check_materialized_complete_f64(out: &[f64]) -> Result<(), TetError> {
    if out.iter().any(|v| v.is_nan()) {
        return Err(TetError::Validation(
            "materialized selection has unset elements (chunk payloads vs selection mismatch)"
                .into(),
        ));
    }
    Ok(())
}

pub(crate) fn materialize_read_plan_f64_le_core(
    mmap: &[u8],
    plan: &ReadPlan,
    max_elements: Option<usize>,
    scatter_fill: ScatterFillF64Fn,
) -> Result<(Vec<f64>, bool, u64), TetError> {
    if matches!(max_elements, Some(0)) {
        return Ok((Vec::new(), false, 0));
    }
    let n = plan.logical_f32_element_count;
    let cap = max_elements.map(|m| m.min(n));
    let (buf_len, truncated) = match cap {
        Some(c) if c < n => (c, true),
        _ => (n, max_elements.is_some_and(|m| m < n)),
    };
    let mut out = vec![f64::NAN; buf_len];
    let total_bytes_read_from_disk = scatter_fill(mmap, plan, &mut out)?;
    check_materialized_complete_f64(&out)?;
    Ok((out, truncated, total_bytes_read_from_disk))
}

/// Decode planned raw `f64` chunk payloads (little-endian) into **logical row-major** order for the
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
pub fn materialize_read_plan_f64_le(
    mmap: &[u8],
    plan: &ReadPlan,
    max_elements: Option<usize>,
) -> Result<(Vec<f64>, bool, u64), TetError> {
    materialize_read_plan_f64_le_core(mmap, plan, max_elements, materialize_scatter_fill_f64)
}

pub(crate) fn materialize_scatter_fill_f64(
    mmap: &[u8],
    plan: &ReadPlan,
    out: &mut [f64],
) -> Result<u64, TetError> {
    validate_read_plan_geometry(plan, out.len())?;
    let mut total_bytes_read_from_disk: u64 = 0;
    for c in &plan.chunks {
        let n = scatter_chunk_into_plan_f64(mmap, plan, c, out)?;
        total_bytes_read_from_disk = total_bytes_read_from_disk
            .checked_add(n)
            .ok_or_else(|| TetError::Validation("total bytes read overflow".into()))?;
    }
    Ok(total_bytes_read_from_disk)
}

/// Spill the full logical selection as row-major `f64` LE to `path` using a file-backed mmap
/// (disk-resident; does not allocate a dense `Vec` in RAM).
///
/// # Errors
///
/// Same validation failures as [`materialize_read_plan_f64_le`], plus I/O or mmap errors on `path`.
pub fn spill_read_plan_f64_le(
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
    let byte_len = f64_le::bytes_from_elem_count(byte_len)
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
    validate_full_read_plan_buffer(plan, out.len())?;
    let total = materialize_scatter_fill_f64(mmap, plan, out)?;
    check_materialized_complete_f64(out)?;
    out_mmap
        .flush()
        .map_err(|e| TetError::Validation(format!("spill mmap flush failed: {e}")))?;
    Ok(total)
}

pub(crate) fn fold_read_plan_scalar_operation_f64(
    mmap: &[u8],
    plan: &ReadPlan,
    max_preview: usize,
    kind: ReductionKind,
    policy: &crate::query::fold::fold_policy::FoldIoPolicy,
) -> Result<FoldPlanOutcome, TetError> {
    if crate::query::fold::parallel_fold::use_parallel_fold(plan, policy) {
        return crate::query::fold::parallel_fold::fold_read_plan_scalar_operation_f64_parallel(
            mmap,
            plan,
            max_preview,
            kind,
            policy.fold_workers,
        );
    }
    let n = plan.logical_f32_element_count;
    let preview_cap = max_preview.min(n);
    let mut f64_preview = vec![f64::NAN; preview_cap];
    let mut total_bytes_read_from_disk: u64 = 0;
    let chunk_order =
        crate::query::fold::fold_policy::chunk_indices_for_fold(plan, policy.sequential_io);

    let operation = match kind {
        ReductionKind::ArgMin | ReductionKind::ArgMax => {
            let mut acc = ArgIndexAccum::default();
            for i in chunk_order {
                let c = &plan.chunks[i];
                let chunk_bytes = visit_planned_chunk_f64(mmap, plan, c, |li, v| {
                    acc.push_f64(li as u64, v, kind);
                    if li < preview_cap {
                        f64_preview[li] = v;
                    }
                    Ok(())
                })?;
                total_bytes_read_from_disk = total_bytes_read_from_disk
                    .checked_add(chunk_bytes)
                    .ok_or_else(|| TetError::Validation("total bytes read overflow".into()))?;
            }
            if acc.is_empty() {
                return Err(TetError::Validation(
                    "operation requires at least one decoded value from the read plan".into(),
                ));
            }
            if preview_cap > 0 && f64_preview.iter().any(|v| v.is_nan()) {
                return Err(TetError::Validation(
                    "materialized selection has unset preview elements".into(),
                ));
            }
            acc.finish_scalar(kind, n).into()
        }
        _ => {
            let mut acc = ValueAccum::default();
            for i in chunk_order {
                let c = &plan.chunks[i];
                let chunk_bytes = visit_planned_chunk_f64(mmap, plan, c, |li, v| {
                    acc.push_f64(v);
                    if li < preview_cap {
                        f64_preview[li] = v;
                    }
                    Ok(())
                })?;
                total_bytes_read_from_disk = total_bytes_read_from_disk
                    .checked_add(chunk_bytes)
                    .ok_or_else(|| TetError::Validation("total bytes read overflow".into()))?;
            }
            if acc.is_empty() {
                return Err(TetError::Validation(
                    "operation requires at least one decoded value from the read plan".into(),
                ));
            }
            if preview_cap > 0 && f64_preview.iter().any(|v| v.is_nan()) {
                return Err(TetError::Validation(
                    "materialized selection has unset preview elements".into(),
                ));
            }
            acc.finish_scalar(kind).into()
        }
    };

    Ok(build_fold_plan_outcome_typed(
        FoldPreviewBuffer::F64(f64_preview),
        max_preview,
        n,
        total_bytes_read_from_disk,
        operation,
    ))
}

pub(crate) fn materialize_into_vec_f64(
    mmap: &[u8],
    plan: &ReadPlan,
) -> Result<(Vec<f64>, u64), TetError> {
    if plan.chunks.len() <= 1 {
        materialize_read_plan_f64_le(mmap, plan, None)
    } else {
        materialize_read_plan_f64_le_parallel(mmap, plan, None)
    }
    .map(|(v, truncated, bytes)| {
        debug_assert!(!truncated);
        (v, bytes)
    })
}

pub(crate) fn preview_from_spill_file_f64(
    path: &std::path::Path,
    cap: usize,
    logical_len: usize,
) -> Result<(Vec<f64>, bool), TetError> {
    use memmap2::Mmap;

    let file = std::fs::File::open(path)
        .map_err(|e| TetError::Validation(format!("temp spill read failed: {e}")))?;
    let mmap = unsafe {
        Mmap::map(&file)
            .map_err(|e| TetError::Validation(format!("temp spill mmap failed: {e}")))?
    };
    let slice: &[f64] = bytemuck::cast_slice(&mmap);
    if slice.len() < cap {
        return Err(TetError::Validation(
            "temp spill shorter than logical selection".into(),
        ));
    }
    Ok((slice[..cap].to_vec(), logical_len > cap))
}
