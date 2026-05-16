use std::collections::BTreeSet;

use crate::catalog::DTYPE_F32;
use crate::query::types::{
    Operation, OperationPreviewFields, QueryExecutionPreview, ReadPlan, TetError,
};

use super::indexing::{coords_from_linear_row_major, linear_rm_index};
use super::materialize::{fold_read_plan_scalar_operation, materialize_read_plan_f32_le};
use super::parallel::materialize_read_plan_f32_le_parallel;
use super::read_plan::shape_product_usize;
use super::reduction::{ReductionKind, ScalarAccum};

fn scalar_reduction_kind(op: &Operation) -> Option<ReductionKind> {
    op.axes().is_empty().then(|| ReductionKind::from(op))
}

/// Decode planned chunks for execution preview (parallel when more than one chunk).
fn materialize_read_plan_f32_le_for_execution(
    mmap: &[u8],
    plan: &ReadPlan,
    max_elements: Option<usize>,
) -> Result<(Vec<f32>, bool, u64), TetError> {
    if plan.chunks.len() <= 1 {
        materialize_read_plan_f32_le(mmap, plan, max_elements)
    } else {
        materialize_read_plan_f32_le_parallel(mmap, plan, max_elements)
    }
}

fn parse_reduction_axes(labels: &[String], ndim: usize) -> Result<Vec<usize>, TetError> {
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

fn fold_count_along(shape: &[u64], axes: &[usize]) -> u64 {
    axes.iter().map(|&d| shape[d]).product()
}

fn reduce_along_axes(
    kind: ReductionKind,
    data: &[f32],
    shape: &[u64],
    axes: &[usize],
) -> Result<(Vec<u64>, Vec<f64>), TetError> {
    let nd = shape.len();
    let axis_set: BTreeSet<usize> = axes.iter().copied().collect();
    let out_shape: Vec<u64> = (0..nd)
        .filter(|d| !axis_set.contains(d))
        .map(|d| shape[d])
        .collect();
    let expected = shape_product_usize(shape)?;
    if data.len() != expected {
        return Err(TetError::Validation(
            "internal: decoded tensor length does not match logical selection shape".into(),
        ));
    }
    let out_len = shape_product_usize(&out_shape)?;
    let mut acc = match kind {
        ReductionKind::Min => vec![f64::INFINITY; out_len],
        ReductionKind::Max => vec![f64::NEG_INFINITY; out_len],
        _ => vec![0.0f64; out_len],
    };
    for (li, &v) in data.iter().enumerate() {
        let coords = coords_from_linear_row_major(li, shape)?;
        let mut out_c = Vec::with_capacity(out_shape.len());
        for (d, cd) in coords.iter().enumerate().take(nd) {
            if !axis_set.contains(&d) {
                out_c.push(*cd);
            }
        }
        let oi = linear_rm_index(&out_c, &out_shape)?;
        let x = f64::from(v);
        match kind {
            ReductionKind::Sum | ReductionKind::Mean => acc[oi] += x,
            ReductionKind::Count => acc[oi] += 1.0,
            ReductionKind::Min => acc[oi] = acc[oi].min(x),
            ReductionKind::Max => acc[oi] = acc[oi].max(x),
        }
    }
    Ok((out_shape, acc))
}

fn validate_operation_tensor(
    values: &[f32],
    shape: &[u64],
    axes: &[String],
) -> Result<(usize, Vec<usize>), TetError> {
    if values.is_empty() {
        return Err(TetError::Validation(
            "operation requires at least one decoded f32 from the read plan".into(),
        ));
    }
    let count = values.len();
    let expected = shape_product_usize(shape)?;
    if count != expected {
        return Err(TetError::Validation(format!(
            "internal: operation tensor has {count} elements but logical shape product is {expected}"
        )));
    }
    let nd = shape.len();
    let axis_indices = parse_reduction_axes(axes, nd)?;
    if axis_indices.len() == nd {
        return Err(TetError::Validation(
            "operation axes list reduces every dimension; use \"axes\": [] for a scalar reduction"
                .into(),
        ));
    }
    Ok((count, axis_indices))
}

fn apply_scalar_operation(kind: ReductionKind, values: &[f32]) -> OperationPreviewFields {
    let mut acc = ScalarAccum::default();
    for &v in values {
        acc.push(v);
    }
    acc.finish(kind).into()
}

fn apply_partial_operation(
    kind: ReductionKind,
    values: &[f32],
    shape: &[u64],
    axis_indices: &[usize],
    element_count: usize,
) -> Result<OperationPreviewFields, TetError> {
    let fc = fold_count_along(shape, axis_indices);
    if fc == 0 {
        return Err(TetError::Validation(
            "internal: zero-sized fold along reduction axes".into(),
        ));
    }
    let (out_shape, reduced) = reduce_along_axes(kind, values, shape, axis_indices)?;
    let mut fields = OperationPreviewFields {
        element_count: Some(element_count),
        reduced_shape: Some(out_shape),
        ..OperationPreviewFields::default()
    };
    match kind {
        ReductionKind::Sum => fields.reduced_sum = Some(reduced),
        ReductionKind::Mean => {
            let fc_div = u32::try_from(fc).map_err(|_| {
                TetError::Validation(
                    "mean reduction: folded element count exceeds `u32::MAX`".into(),
                )
            })?;
            let fc_f = f64::from(fc_div);
            fields.reduced_mean = Some(reduced.iter().map(|s| s / fc_f).collect());
        }
        ReductionKind::Min => fields.reduced_min = Some(reduced),
        ReductionKind::Max => fields.reduced_max = Some(reduced),
        ReductionKind::Count => fields.reduced_count = Some(reduced),
    }
    Ok(fields)
}

fn apply_operation(
    op: &Operation,
    values: &[f32],
    shape: &[u64],
) -> Result<OperationPreviewFields, TetError> {
    let kind = ReductionKind::from(op);
    let (count, axis_indices) = validate_operation_tensor(values, shape, op.axes())?;
    if axis_indices.is_empty() {
        Ok(apply_scalar_operation(kind, values))
    } else {
        apply_partial_operation(kind, values, shape, &axis_indices, count)
    }
}

pub(super) fn build_execution_preview(
    mmap: &[u8],
    plan: &ReadPlan,
    dtype: u32,
    operation: Option<&Operation>,
    max_f32: usize,
) -> Result<QueryExecutionPreview, TetError> {
    if dtype != DTYPE_F32 {
        return Err(TetError::Validation(
            "f32 preview requires dataset dtype f32 (DTYPE_F32 = 1)".into(),
        ));
    }
    match operation {
        None => {
            let (f32_preview, f32_preview_truncated, total_bytes_read_from_disk) =
                materialize_read_plan_f32_le_for_execution(mmap, plan, Some(max_f32))?;
            Ok(QueryExecutionPreview::decode_preview(
                total_bytes_read_from_disk,
                f32_preview,
                f32_preview_truncated,
            ))
        }
        Some(op) => {
            if let Some(kind) = scalar_reduction_kind(op) {
                let folded = fold_read_plan_scalar_operation(mmap, plan, max_f32, kind)?;
                return Ok(QueryExecutionPreview::with_operation(
                    folded.total_bytes_read_from_disk,
                    folded.f32_preview,
                    folded.f32_preview_truncated,
                    folded.scalar.into(),
                ));
            }
            let (full, _, total_bytes_read_from_disk) =
                materialize_read_plan_f32_le_for_execution(mmap, plan, None)?;
            let operation = apply_operation(op, &full, &plan.logical_selection_shape)?;
            let f32_preview_truncated = full.len() > max_f32;
            let f32_preview: Vec<f32> = full.iter().take(max_f32).copied().collect();
            Ok(QueryExecutionPreview::with_operation(
                total_bytes_read_from_disk,
                f32_preview,
                f32_preview_truncated,
                operation,
            ))
        }
    }
}
