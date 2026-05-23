//! Chunk-streaming partial-axis reductions (no full logical tensor allocation).

use std::collections::BTreeSet;

use crate::query::types::{OperationPreviewFields, ReadPlan, TetError};

use super::chunk_decode::visit_planned_chunk;
use super::fold::{FoldPlanOutcome, build_fold_plan_outcome, validate_fold_preview};
use super::indexing::{coords_from_linear_row_major, linear_rm_index};
use super::read_plan::shape_product_usize;
use super::reduction::{ArgIndexAccum, ReductionKind, ValueAccum};

fn parse_axis_indices(labels: &[String], ndim: usize) -> Result<Vec<usize>, TetError> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for label in labels {
        let v: usize = label
            .parse()
            .map_err(|_| TetError::Validation(format!("invalid operation axis index {label:?}")))?;
        if v >= ndim {
            return Err(TetError::Validation(format!(
                "operation axis index {v} out of range for rank {ndim}"
            )));
        }
        if !seen.insert(v) {
            return Err(TetError::Validation(format!(
                "duplicate operation axis index {v}"
            )));
        }
        out.push(v);
    }
    out.sort_unstable();
    Ok(out)
}

fn out_shape(shape: &[u64], axis_indices: &[usize]) -> Vec<u64> {
    let axis_set: BTreeSet<usize> = axis_indices.iter().copied().collect();
    shape
        .iter()
        .enumerate()
        .filter(|(d, _)| !axis_set.contains(d))
        .map(|(_, &e)| e)
        .collect()
}

fn reduced_index(
    coords: &[usize],
    axis_set: &BTreeSet<usize>,
    out_shape: &[u64],
) -> Result<usize, TetError> {
    let mut out_c = Vec::new();
    for (d, &c) in coords.iter().enumerate() {
        if !axis_set.contains(&d) {
            out_c.push(c);
        }
    }
    linear_rm_index(&out_c, out_shape)
}

fn fiber_linear_index(
    coords: &[usize],
    axis_indices: &[usize],
    shape: &[u64],
) -> Result<usize, TetError> {
    let rshape: Vec<u64> = axis_indices.iter().map(|&d| shape[d]).collect();
    let rc: Vec<usize> = axis_indices.iter().map(|&d| coords[d]).collect();
    linear_rm_index(&rc, &rshape)
}

/// Stream a **partial-axis** reduction without allocating the full logical tensor.
pub(crate) fn fold_read_plan_partial_operation(
    mmap: &[u8],
    plan: &ReadPlan,
    max_f32: usize,
    kind: ReductionKind,
    axis_labels: &[String],
) -> Result<FoldPlanOutcome, TetError> {
    let shape = &plan.logical_selection_shape;
    let ndim = shape.len();
    let axis_indices = parse_axis_indices(axis_labels, ndim)?;
    if axis_indices.is_empty() {
        return Err(TetError::Validation(
            "internal: partial fold requires non-empty axes".into(),
        ));
    }
    if axis_indices.len() == ndim {
        return Err(TetError::Validation(
            "operation axes list reduces every dimension; use \"axes\": [] for a scalar reduction"
                .into(),
        ));
    }
    let out_sh = out_shape(shape, &axis_indices);
    let out_len = shape_product_usize(&out_sh)?;
    let axis_set: BTreeSet<usize> = axis_indices.iter().copied().collect();

    let n = plan.logical_f32_element_count;
    let preview_cap = max_f32.min(n);
    let mut preview = vec![f32::NAN; preview_cap];
    let mut saw_any = false;
    let mut total_bytes_read_from_disk: u64 = 0;

    let operation = match kind {
        ReductionKind::ArgMin | ReductionKind::ArgMax => {
            let mut cells = vec![ArgIndexAccum::default(); out_len];
            for c in &plan.chunks {
                let chunk_bytes = visit_planned_chunk(mmap, plan, c, |li, v| {
                    saw_any = true;
                    if li < preview_cap {
                        preview[li] = v;
                    }
                    let coords = coords_from_linear_row_major(li, shape)?;
                    let oi = reduced_index(&coords, &axis_set, &out_sh)?;
                    let fi = fiber_linear_index(&coords, &axis_indices, shape)? as u64;
                    cells[oi].push(fi, v, kind);
                    Ok(())
                })?;
                total_bytes_read_from_disk = total_bytes_read_from_disk
                    .checked_add(chunk_bytes)
                    .ok_or_else(|| TetError::Validation("total bytes read overflow".into()))?;
            }
            validate_fold_preview(saw_any, &preview, preview_cap)?;
            partial_arg_fields(kind, n, &out_sh, &cells)
        }
        _ => {
            let mut cells = vec![ValueAccum::default(); out_len];
            for c in &plan.chunks {
                let chunk_bytes = visit_planned_chunk(mmap, plan, c, |li, v| {
                    saw_any = true;
                    if li < preview_cap {
                        preview[li] = v;
                    }
                    let coords = coords_from_linear_row_major(li, shape)?;
                    let oi = reduced_index(&coords, &axis_set, &out_sh)?;
                    cells[oi].push(v);
                    Ok(())
                })?;
                total_bytes_read_from_disk = total_bytes_read_from_disk
                    .checked_add(chunk_bytes)
                    .ok_or_else(|| TetError::Validation("total bytes read overflow".into()))?;
            }
            validate_fold_preview(saw_any, &preview, preview_cap)?;
            let reduced: Vec<f64> = cells.iter().map(|c| c.finish_f64(kind)).collect();
            partial_fields(kind, n, &out_sh, &reduced, &cells)
        }
    };

    Ok(build_fold_plan_outcome(
        preview,
        max_f32,
        n,
        total_bytes_read_from_disk,
        operation,
    ))
}

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
