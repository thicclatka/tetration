use std::collections::BTreeSet;

use crate::catalog::DTYPE_F32;
use crate::query::types::{Operation, QueryExecutionPreview, ReadPlan, TetError};

use super::indexing::{coords_from_linear_row_major, linear_rm_index};
use super::materialize::materialize_read_plan_f32_le;
use super::parallel::materialize_read_plan_f32_le_parallel;
use super::read_plan::shape_product_usize;

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

fn reduce_sum_along_axes(
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
    let mut acc = vec![0.0f64; out_len];
    for (li, &v) in data.iter().enumerate() {
        let coords = coords_from_linear_row_major(li, shape)?;
        let mut out_c = Vec::with_capacity(out_shape.len());
        for (d, cd) in coords.iter().enumerate().take(nd) {
            if !axis_set.contains(&d) {
                out_c.push(*cd);
            }
        }
        let oi = linear_rm_index(&out_c, &out_shape)?;
        acc[oi] += f64::from(v);
    }
    Ok((out_shape, acc))
}

struct OperationAgg {
    element_count: usize,
    sum_scalar: Option<f64>,
    mean_scalar: Option<f64>,
    reduced_shape: Option<Vec<u64>>,
    reduced_sum: Option<Vec<f64>>,
    reduced_mean: Option<Vec<f64>>,
}

fn apply_operation(
    op: &Operation,
    values: &[f32],
    shape: &[u64],
) -> Result<OperationAgg, TetError> {
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
    let axes = match op {
        Operation::Sum { axes } | Operation::Mean { axes } => axes,
    };
    let axis_indices = parse_reduction_axes(axes, nd)?;
    if axis_indices.len() == nd {
        return Err(TetError::Validation(
            "operation axes list reduces every dimension; use \"axes\": [] for a scalar reduction"
                .into(),
        ));
    }
    if axis_indices.is_empty() {
        return match op {
            Operation::Sum { .. } => {
                let s: f64 = values.iter().map(|&x| f64::from(x)).sum();
                Ok(OperationAgg {
                    element_count: count,
                    sum_scalar: Some(s),
                    mean_scalar: None,
                    reduced_shape: None,
                    reduced_sum: None,
                    reduced_mean: None,
                })
            }
            Operation::Mean { .. } => {
                let mut mean = 0.0f64;
                let mut k = 0.0f64;
                for &x in values {
                    k += 1.0;
                    mean += (f64::from(x) - mean) / k;
                }
                Ok(OperationAgg {
                    element_count: count,
                    sum_scalar: None,
                    mean_scalar: Some(mean),
                    reduced_shape: None,
                    reduced_sum: None,
                    reduced_mean: None,
                })
            }
        };
    }
    let fc = fold_count_along(shape, &axis_indices);
    if fc == 0 {
        return Err(TetError::Validation(
            "internal: zero-sized fold along reduction axes".into(),
        ));
    }
    let (out_shape, sums) = reduce_sum_along_axes(values, shape, &axis_indices)?;
    match op {
        Operation::Sum { .. } => Ok(OperationAgg {
            element_count: count,
            sum_scalar: None,
            mean_scalar: None,
            reduced_shape: Some(out_shape),
            reduced_sum: Some(sums),
            reduced_mean: None,
        }),
        Operation::Mean { .. } => {
            let fc_div = u32::try_from(fc).map_err(|_| {
                TetError::Validation(
                    "mean reduction: folded element count exceeds `u32::MAX`".into(),
                )
            })?;
            let fc_f = f64::from(fc_div);
            let means: Vec<f64> = sums.iter().map(|s| s / fc_f).collect();
            Ok(OperationAgg {
                element_count: count,
                sum_scalar: None,
                mean_scalar: None,
                reduced_shape: Some(out_shape),
                reduced_sum: None,
                reduced_mean: Some(means),
            })
        }
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
            Ok(QueryExecutionPreview {
                total_bytes_read_from_disk,
                f32_preview,
                f32_preview_truncated,
                operation_element_count: None,
                operation_sum: None,
                operation_mean: None,
                operation_reduced_shape: None,
                operation_reduced_sum: None,
                operation_reduced_mean: None,
            })
        }
        Some(op) => {
            let (full, _, total_bytes_read_from_disk) =
                materialize_read_plan_f32_le_for_execution(mmap, plan, None)?;
            let agg = apply_operation(op, &full, &plan.logical_selection_shape)?;
            let f32_preview_truncated = full.len() > max_f32;
            let f32_preview: Vec<f32> = full.iter().take(max_f32).copied().collect();
            Ok(QueryExecutionPreview {
                total_bytes_read_from_disk,
                f32_preview,
                f32_preview_truncated,
                operation_element_count: Some(agg.element_count),
                operation_sum: agg.sum_scalar,
                operation_mean: agg.mean_scalar,
                operation_reduced_shape: agg.reduced_shape,
                operation_reduced_sum: agg.reduced_sum,
                operation_reduced_mean: agg.reduced_mean,
            })
        }
    }
}
