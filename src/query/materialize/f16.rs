//! `f16` logical materialize, spill, and scalar fold.

use std::path::Path;

use half::f16;

use crate::query::{
    decode::chunk_decode,
    fold,
    types::{ReadPlan, TetError},
};
use crate::utils::f16_le;

use super::parallel::materialize_read_plan_f16_le_parallel;
use super::shared::{
    check_materialized_nan_slice, materialize_into_vec_dispatch, materialize_read_plan_pod_le_core,
    scatter_fill_chunks, spill_byte_len_from_elem_count, spill_read_plan_pod_le_impl,
};

type ScatterFillF16Fn = fn(&[u8], &ReadPlan, &mut [f16]) -> Result<u64, TetError>;

fn check_materialized_complete_f16(out: &[f16]) -> Result<(), TetError> {
    check_materialized_nan_slice(out, f16::is_nan)
}

pub(crate) fn materialize_read_plan_f16_le_core(
    mmap: &[u8],
    plan: &ReadPlan,
    max_elements: Option<usize>,
    scatter_fill: ScatterFillF16Fn,
) -> Result<(Vec<f16>, bool, u64), TetError> {
    materialize_read_plan_pod_le_core(
        mmap,
        plan,
        max_elements,
        scatter_fill,
        f16::NAN,
        check_materialized_complete_f16,
    )
}

pub fn materialize_read_plan_f16_le(
    mmap: &[u8],
    plan: &ReadPlan,
    max_elements: Option<usize>,
) -> Result<(Vec<f16>, bool, u64), TetError> {
    materialize_read_plan_f16_le_core(mmap, plan, max_elements, materialize_scatter_fill_f16)
}

pub(crate) fn materialize_scatter_fill_f16(
    mmap: &[u8],
    plan: &ReadPlan,
    out: &mut [f16],
) -> Result<u64, TetError> {
    scatter_fill_chunks(mmap, plan, out, chunk_decode::scatter_chunk_into_plan_f16)
}

pub fn spill_read_plan_f16_le(mmap: &[u8], plan: &ReadPlan, path: &Path) -> Result<u64, TetError> {
    let byte_len = spill_byte_len_from_elem_count(
        plan.logical_f32_element_count,
        f16_le::bytes_from_elem_count,
    )?;
    spill_read_plan_pod_le_impl(
        mmap,
        plan,
        path,
        byte_len,
        materialize_scatter_fill_f16,
        check_materialized_complete_f16,
    )
}

pub(crate) fn fold_read_plan_scalar_operation_f16(
    mmap: &[u8],
    plan: &ReadPlan,
    max_preview: usize,
    kind: fold::reduction::ReductionKind,
    policy: &crate::query::fold::fold_policy::FoldIoPolicy,
) -> Result<fold::FoldPlanOutcome, TetError> {
    if crate::query::fold::parallel::use_parallel_fold(plan, policy) {
        return crate::query::fold::parallel::fold_read_plan_scalar_operation_f16_parallel(
            mmap,
            plan,
            max_preview,
            kind,
            policy.fold_workers,
        );
    }
    let n = plan.logical_f32_element_count;
    let preview_cap = max_preview.min(n);
    let mut f16_preview = vec![f16::NAN; preview_cap];
    let mut total_bytes_read_from_disk: u64 = 0;
    let chunk_order =
        crate::query::fold::fold_policy::chunk_indices_for_fold(plan, policy.sequential_io);

    let operation = match kind {
        fold::reduction::ReductionKind::ArgMin | fold::reduction::ReductionKind::ArgMax => {
            let mut acc = fold::reduction::ArgIndexAccum::default();
            for i in chunk_order {
                let c = &plan.chunks[i];
                let chunk_bytes = chunk_decode::visit_planned_chunk_f16(mmap, plan, c, |li, v| {
                    acc.push_f64(li as u64, f64::from(v), kind);
                    if li < preview_cap {
                        f16_preview[li] = v;
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
            if preview_cap > 0 && f16_preview.iter().any(|v| v.is_nan()) {
                return Err(TetError::Validation(
                    "materialized selection has unset preview elements".into(),
                ));
            }
            acc.finish_scalar(kind, n).into()
        }
        _ => {
            let mut acc = fold::reduction::ValueAccum::default();
            for i in chunk_order {
                let c = &plan.chunks[i];
                let chunk_bytes = chunk_decode::visit_planned_chunk_f16(mmap, plan, c, |li, v| {
                    acc.push_f64(f64::from(v));
                    if li < preview_cap {
                        f16_preview[li] = v;
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
            if preview_cap > 0 && f16_preview.iter().any(|v| v.is_nan()) {
                return Err(TetError::Validation(
                    "materialized selection has unset preview elements".into(),
                ));
            }
            acc.finish_scalar(kind).into()
        }
    };

    Ok(fold::shared::build_fold_plan_outcome_typed(
        fold::shared::FoldPreviewBuffer::F16(f16_preview),
        max_preview,
        n,
        total_bytes_read_from_disk,
        operation,
    ))
}

pub(crate) fn materialize_into_vec_f16(
    mmap: &[u8],
    plan: &ReadPlan,
) -> Result<(Vec<f16>, u64), TetError> {
    materialize_into_vec_dispatch(
        mmap,
        plan,
        materialize_read_plan_f16_le,
        materialize_read_plan_f16_le_parallel,
    )
}

pub(crate) fn preview_from_materialized_f16(
    backing: &super::types::LogicalF16Backing,
    logical_len: usize,
    max: usize,
) -> Result<(Vec<f16>, bool), TetError> {
    use super::shared::preview_from_backing_in_memory;

    match backing {
        super::types::LogicalF16Backing::InMemory(v) => {
            Ok(preview_from_backing_in_memory(v, logical_len, max))
        }
        super::types::LogicalF16Backing::TempSpill(temp) => {
            super::shared::preview_from_spill_file_pod(
                temp.path(),
                max.min(logical_len),
                logical_len,
            )
        }
    }
}

pub(crate) fn preview_from_spill_file_f16(
    path: &Path,
    cap: usize,
    logical_len: usize,
) -> Result<(Vec<f16>, bool), TetError> {
    super::shared::preview_from_spill_file_pod(path, cap, logical_len)
}
