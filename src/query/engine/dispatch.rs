//! Shared dtype-dispatch helpers for query execution.

use std::path::Path;

use crate::query::types::{ReadPlan, TetError};
use crate::utils::dtype::ElementDtype;

use super::fold::FoldPlanOutcome;
use super::materialize::{
    DecodePreviewBundle, materialize_read_plan_f32_le, materialize_read_plan_f64_le,
    preview_from_spill_export_file, spill_read_plan_f32_le, spill_read_plan_f64_le,
};
use super::materialize_int::{
    fold_read_plan_scalar_operation_int, materialize_read_plan_i32_le,
    materialize_read_plan_i64_le, spill_read_plan_int_le,
};
use super::parallel::{
    materialize_read_plan_f32_le_parallel, materialize_read_plan_f64_le_parallel,
    materialize_read_plan_i32_le_parallel, materialize_read_plan_i64_le_parallel,
};
use super::partial_fold::{
    fold_read_plan_partial_operation, fold_read_plan_partial_operation_f64,
    fold_read_plan_partial_operation_int,
};
use super::reduction::ReductionKind;

pub(crate) fn accumulate_chunk_read_bytes(
    total: &mut u64,
    chunk_bytes: u64,
) -> Result<(), TetError> {
    *total = total
        .checked_add(chunk_bytes)
        .ok_or_else(|| TetError::Validation("total bytes read overflow".into()))?;
    Ok(())
}

pub(crate) fn materialize_for_execution(
    mmap: &[u8],
    plan: &ReadPlan,
    max_elements: Option<usize>,
    dtype: ElementDtype,
) -> Result<(DecodePreviewBundle, u64), TetError> {
    let parallel = plan.chunks.len() > 1;
    match dtype {
        ElementDtype::F32 => {
            let (p, t, bytes) = if parallel {
                materialize_read_plan_f32_le_parallel(mmap, plan, max_elements)?
            } else {
                materialize_read_plan_f32_le(mmap, plan, max_elements)?
            };
            Ok((
                DecodePreviewBundle {
                    f32: p,
                    f32_truncated: t,
                    ..DecodePreviewBundle::empty()
                },
                bytes,
            ))
        }
        ElementDtype::F64 => {
            let (p, t, bytes) = if parallel {
                materialize_read_plan_f64_le_parallel(mmap, plan, max_elements)?
            } else {
                materialize_read_plan_f64_le(mmap, plan, max_elements)?
            };
            Ok((
                DecodePreviewBundle {
                    f64: p,
                    f64_truncated: t,
                    ..DecodePreviewBundle::empty()
                },
                bytes,
            ))
        }
        ElementDtype::I32 => {
            let (p, t, bytes) = if parallel {
                materialize_read_plan_i32_le_parallel(mmap, plan, max_elements)?
            } else {
                materialize_read_plan_i32_le(mmap, plan, max_elements)?
            };
            Ok((
                DecodePreviewBundle {
                    i32: p,
                    i32_truncated: t,
                    ..DecodePreviewBundle::empty()
                },
                bytes,
            ))
        }
        ElementDtype::I64 => {
            let (p, t, bytes) = if parallel {
                materialize_read_plan_i64_le_parallel(mmap, plan, max_elements)?
            } else {
                materialize_read_plan_i64_le(mmap, plan, max_elements)?
            };
            Ok((
                DecodePreviewBundle {
                    i64: p,
                    i64_truncated: t,
                    ..DecodePreviewBundle::empty()
                },
                bytes,
            ))
        }
    }
}

pub(crate) fn spill_full_selection(
    mmap: &[u8],
    plan: &ReadPlan,
    path: &Path,
    dtype: ElementDtype,
) -> Result<u64, TetError> {
    match dtype {
        ElementDtype::F32 => spill_read_plan_f32_le(mmap, plan, path),
        ElementDtype::F64 => spill_read_plan_f64_le(mmap, plan, path),
        ElementDtype::I32 | ElementDtype::I64 => spill_read_plan_int_le(mmap, plan, path, dtype),
    }
}

pub(crate) fn spill_export_preview(
    path: &Path,
    logical_len: usize,
    max_preview: usize,
    dtype: ElementDtype,
) -> Result<DecodePreviewBundle, TetError> {
    preview_from_spill_export_file(path, logical_len, max_preview, dtype)
}

pub(crate) fn scalar_fold(
    mmap: &[u8],
    plan: &ReadPlan,
    max_preview: usize,
    kind: ReductionKind,
    dtype: ElementDtype,
) -> Result<FoldPlanOutcome, TetError> {
    match dtype {
        ElementDtype::F32 => {
            super::materialize::fold_read_plan_scalar_operation(mmap, plan, max_preview, kind)
        }
        ElementDtype::F64 => {
            super::materialize::fold_read_plan_scalar_operation_f64(mmap, plan, max_preview, kind)
        }
        ElementDtype::I32 | ElementDtype::I64 => {
            fold_read_plan_scalar_operation_int(mmap, plan, max_preview, kind, dtype)
        }
    }
}

pub(crate) fn partial_fold(
    mmap: &[u8],
    plan: &ReadPlan,
    max_preview: usize,
    kind: ReductionKind,
    axis_labels: &[String],
    dtype: ElementDtype,
) -> Result<FoldPlanOutcome, TetError> {
    match dtype {
        ElementDtype::F32 => {
            fold_read_plan_partial_operation(mmap, plan, max_preview, kind, axis_labels)
        }
        ElementDtype::F64 => {
            fold_read_plan_partial_operation_f64(mmap, plan, max_preview, kind, axis_labels)
        }
        ElementDtype::I32 | ElementDtype::I64 => {
            fold_read_plan_partial_operation_int(mmap, plan, max_preview, kind, axis_labels, dtype)
        }
    }
}
