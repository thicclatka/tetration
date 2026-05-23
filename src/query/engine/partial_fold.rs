//! Chunk-streaming partial-axis reductions (no full logical tensor allocation).

#![allow(clippy::too_many_lines, clippy::too_many_arguments)]

use crate::query::types::{OperationPreviewFields, ReadPlan, TetError};

use super::chunk_decode::{visit_planned_chunk, visit_planned_chunk_f64};
use super::fold::{FoldPlanOutcome, build_fold_plan_outcome, validate_fold_preview};
use super::indexing::coords_from_linear_row_major;
use super::partial_geometry::{partial_axis_layout, reduced_index};
use super::reduction::{ArgIndexAccum, ReductionKind, ValueAccum};

fn partial_arg_fields(
    kind: ReductionKind,
    element_count: usize,
    out_shape: &[u64],
    cells: &[ArgIndexAccum],
) -> OperationPreviewFields {
    let indices: Vec<u64> = cells.iter().map(ArgIndexAccum::index).collect();
    let mut fields = OperationPreviewFields {
        element_count: Some(element_count),
        reduced_shape: Some(out_shape.to_vec()),
        ..OperationPreviewFields::default()
    };
    match kind {
        ReductionKind::ArgMin => fields.reduced_argmin = Some(indices),
        ReductionKind::ArgMax => fields.reduced_argmax = Some(indices),
        _ => {}
    }
    fields
}

fn partial_fields(
    kind: ReductionKind,
    element_count: usize,
    out_shape: &[u64],
    reduced: &[f64],
    cells: &[ValueAccum],
) -> OperationPreviewFields {
    let mut fields = OperationPreviewFields {
        element_count: Some(element_count),
        reduced_shape: Some(out_shape.to_vec()),
        ..OperationPreviewFields::default()
    };
    match kind {
        ReductionKind::Sum => fields.reduced_sum = Some(reduced.to_vec()),
        ReductionKind::Mean => fields.reduced_mean = Some(reduced.to_vec()),
        ReductionKind::Min => fields.reduced_min = Some(reduced.to_vec()),
        ReductionKind::Max => fields.reduced_max = Some(reduced.to_vec()),
        ReductionKind::Count => fields.reduced_count = Some(reduced.to_vec()),
        ReductionKind::Var => fields.reduced_var = Some(reduced.to_vec()),
        ReductionKind::Std => fields.reduced_std = Some(reduced.to_vec()),
        ReductionKind::Product => fields.reduced_product = Some(reduced.to_vec()),
        ReductionKind::NormL1 => fields.reduced_norm_l1 = Some(reduced.to_vec()),
        ReductionKind::NormL2 => fields.reduced_norm_l2 = Some(reduced.to_vec()),
        ReductionKind::AllFinite => {
            fields.reduced_all_finite = Some(cells.iter().map(|c| c.finish_bool(kind)).collect());
        }
        ReductionKind::AnyNan => {
            fields.reduced_any_nan = Some(cells.iter().map(|c| c.finish_bool(kind)).collect());
        }
        ReductionKind::ArgMin | ReductionKind::ArgMax => {
            unreachable!("argmin/argmax use partial_arg_fields")
        }
    }
    fields
}

/// Stream a **partial-axis** reduction without allocating the full logical tensor (f32).
pub(crate) fn fold_read_plan_partial_operation(
    mmap: &[u8],
    plan: &ReadPlan,
    max_f32: usize,
    kind: ReductionKind,
    axis_labels: &[String],
) -> Result<FoldPlanOutcome, TetError> {
    fold_read_plan_partial_operation_impl(mmap, plan, max_f32, kind, axis_labels, false)
}

/// Stream a **partial-axis** reduction without allocating the full logical tensor (f64).
pub(crate) fn fold_read_plan_partial_operation_f64(
    mmap: &[u8],
    plan: &ReadPlan,
    max_preview: usize,
    kind: ReductionKind,
    axis_labels: &[String],
) -> Result<FoldPlanOutcome, TetError> {
    fold_read_plan_partial_operation_impl(mmap, plan, max_preview, kind, axis_labels, true)
}

fn fold_read_plan_partial_operation_impl(
    mmap: &[u8],
    plan: &ReadPlan,
    max_preview: usize,
    kind: ReductionKind,
    axis_labels: &[String],
    f64_path: bool,
) -> Result<FoldPlanOutcome, TetError> {
    let layout = partial_axis_layout(plan, axis_labels)?;
    let shape = &plan.logical_selection_shape;
    let n = plan.logical_f32_element_count;
    let preview_cap = max_preview.min(n);
    let mut f32_preview = vec![f32::NAN; preview_cap];
    let mut f64_preview = vec![f64::NAN; preview_cap];
    let mut saw_any = false;
    let mut total_bytes_read_from_disk: u64 = 0;

    let operation = if f64_path {
        run_partial_f64(
            mmap,
            plan,
            kind,
            &layout,
            shape,
            n,
            preview_cap,
            &mut f64_preview,
            &mut saw_any,
            &mut total_bytes_read_from_disk,
        )?
    } else {
        run_partial_f32(
            mmap,
            plan,
            kind,
            &layout,
            shape,
            n,
            preview_cap,
            &mut f32_preview,
            &mut saw_any,
            &mut total_bytes_read_from_disk,
        )?
    };

    if f64_path {
        validate_fold_preview_f64(saw_any, &f64_preview, preview_cap)?;
        Ok(FoldPlanOutcome {
            f32_preview: Vec::new(),
            f64_preview: if max_preview == 0 {
                Vec::new()
            } else {
                f64_preview
            },
            f32_preview_truncated: false,
            f64_preview_truncated: n > max_preview,
            total_bytes_read_from_disk,
            operation,
        })
    } else {
        validate_fold_preview(saw_any, &f32_preview, preview_cap)?;
        Ok(build_fold_plan_outcome(
            f32_preview,
            max_preview,
            n,
            total_bytes_read_from_disk,
            operation,
        ))
    }
}

fn run_partial_f32(
    mmap: &[u8],
    plan: &ReadPlan,
    kind: ReductionKind,
    layout: &super::partial_geometry::PartialAxisLayout,
    shape: &[u64],
    n: usize,
    preview_cap: usize,
    preview: &mut [f32],
    saw_any: &mut bool,
    total_bytes: &mut u64,
) -> Result<OperationPreviewFields, TetError> {
    match kind {
        ReductionKind::ArgMin | ReductionKind::ArgMax => {
            let mut cells = vec![ArgIndexAccum::default(); layout.out_len];
            for c in &plan.chunks {
                let chunk_bytes = visit_planned_chunk(mmap, plan, c, |li, v| {
                    *saw_any = true;
                    if li < preview_cap {
                        preview[li] = v;
                    }
                    let coords = coords_from_linear_row_major(li, shape)?;
                    let oi = reduced_index(&coords, &layout.axis_set, &layout.out_shape)?;
                    let fi = super::partial_geometry::fiber_linear_index(
                        &coords,
                        &layout.axis_indices,
                        shape,
                    )? as u64;
                    cells[oi].push(fi, v, kind);
                    Ok(())
                })?;
                *total_bytes = total_bytes
                    .checked_add(chunk_bytes)
                    .ok_or_else(|| TetError::Validation("total bytes read overflow".into()))?;
            }
            Ok(partial_arg_fields(kind, n, &layout.out_shape, &cells))
        }
        _ => {
            let mut cells = vec![ValueAccum::default(); layout.out_len];
            for c in &plan.chunks {
                let chunk_bytes = visit_planned_chunk(mmap, plan, c, |li, v| {
                    *saw_any = true;
                    if li < preview_cap {
                        preview[li] = v;
                    }
                    let coords = coords_from_linear_row_major(li, shape)?;
                    let oi = reduced_index(&coords, &layout.axis_set, &layout.out_shape)?;
                    cells[oi].push(v);
                    Ok(())
                })?;
                *total_bytes = total_bytes
                    .checked_add(chunk_bytes)
                    .ok_or_else(|| TetError::Validation("total bytes read overflow".into()))?;
            }
            let reduced: Vec<f64> = cells.iter().map(|c| c.finish_f64(kind)).collect();
            Ok(partial_fields(kind, n, &layout.out_shape, &reduced, &cells))
        }
    }
}

fn run_partial_f64(
    mmap: &[u8],
    plan: &ReadPlan,
    kind: ReductionKind,
    layout: &super::partial_geometry::PartialAxisLayout,
    shape: &[u64],
    n: usize,
    preview_cap: usize,
    preview: &mut [f64],
    saw_any: &mut bool,
    total_bytes: &mut u64,
) -> Result<OperationPreviewFields, TetError> {
    match kind {
        ReductionKind::ArgMin | ReductionKind::ArgMax => {
            let mut cells = vec![ArgIndexAccum::default(); layout.out_len];
            for c in &plan.chunks {
                let chunk_bytes = visit_planned_chunk_f64(mmap, plan, c, |li, v| {
                    *saw_any = true;
                    if li < preview_cap {
                        preview[li] = v;
                    }
                    let coords = coords_from_linear_row_major(li, shape)?;
                    let oi = reduced_index(&coords, &layout.axis_set, &layout.out_shape)?;
                    let fi = super::partial_geometry::fiber_linear_index(
                        &coords,
                        &layout.axis_indices,
                        shape,
                    )? as u64;
                    cells[oi].push_f64(fi, v, kind);
                    Ok(())
                })?;
                *total_bytes = total_bytes
                    .checked_add(chunk_bytes)
                    .ok_or_else(|| TetError::Validation("total bytes read overflow".into()))?;
            }
            Ok(partial_arg_fields(kind, n, &layout.out_shape, &cells))
        }
        _ => {
            let mut cells = vec![ValueAccum::default(); layout.out_len];
            for c in &plan.chunks {
                let chunk_bytes = visit_planned_chunk_f64(mmap, plan, c, |li, v| {
                    *saw_any = true;
                    if li < preview_cap {
                        preview[li] = v;
                    }
                    let coords = coords_from_linear_row_major(li, shape)?;
                    let oi = reduced_index(&coords, &layout.axis_set, &layout.out_shape)?;
                    cells[oi].push_f64(v);
                    Ok(())
                })?;
                *total_bytes = total_bytes
                    .checked_add(chunk_bytes)
                    .ok_or_else(|| TetError::Validation("total bytes read overflow".into()))?;
            }
            let reduced: Vec<f64> = cells.iter().map(|c| c.finish_f64(kind)).collect();
            Ok(partial_fields(kind, n, &layout.out_shape, &reduced, &cells))
        }
    }
}

fn validate_fold_preview_f64(
    saw_any: bool,
    preview: &[f64],
    preview_cap: usize,
) -> Result<(), TetError> {
    if !saw_any {
        return Err(TetError::Validation(
            "operation requires at least one decoded value from the read plan".into(),
        ));
    }
    if preview_cap > 0 && preview.iter().any(|v| v.is_nan()) {
        return Err(TetError::Validation(
            "materialized selection has unset preview elements (chunk payloads vs selection mismatch)"
                .into(),
        ));
    }
    Ok(())
}
