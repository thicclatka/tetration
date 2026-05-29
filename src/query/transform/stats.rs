//! Pass-1 statistics for element-wise transforms.
//!
//! [`collect_transform_stats`] streams planned chunks (or dispatches standard fold
//! reductions) to build per-cell stat vectors used in pass 2 ([`super::apply`]).
//! Global transforms (`axes: []`) use length-1 vectors; partial-axis transforms
//! use one entry per reduced output cell (`layout.out_len`).

use std::path::Path;

use crate::query::decode::chunk_decode::{visit_planned_chunk, visit_planned_chunk_f64};
use crate::query::dispatch::{self, accumulate_chunk_read_bytes};
use crate::query::fold::{
    fold_policy::FoldIoPolicy,
    partial_geometry::{self, PartialAxisLayout},
    reduction::ReductionKind,
};
use crate::query::types::{Operation, OperationPreviewFields, ReadPlan, TetError};
use crate::utils::dtype::ElementDtype;

use super::TransformMethod;

const ERR_TRANSFORM_FLOAT: &str = "transform requires f32 or f64 datasets";

/// Per-cell statistics gathered in pass 1 (indexed via [`partial_geometry::reduced_cell_index_or_global`]).
#[derive(Debug, Clone)]
pub(crate) struct TransformStats {
    pub method: TransformMethod,
    /// Set after layout resolution; `None` for global (`axes: []`) transforms.
    pub layout: Option<PartialAxisLayout>,
    /// Mean per cell (`center`, `zscore`).
    pub mean: Vec<f64>,
    /// Population std per cell (`zscore`, `scale`), `ddof = 0`.
    pub std: Vec<f64>,
    /// Min per cell (`minmax`, `log1p`, `sqrt` shift).
    pub min: Vec<f64>,
    /// Max per cell (`minmax`, `softmax` stabilization).
    pub max: Vec<f64>,
    /// L1 norm per cell (`l1`).
    pub norm_l1: Vec<f64>,
    /// L2 norm per cell (`l2`).
    pub norm_l2: Vec<f64>,
    /// `sum(exp(x - max))` per cell after pass-1 max fold (`softmax`).
    pub sum_exp: Vec<f64>,
}

impl TransformStats {
    fn cell_len(&self) -> usize {
        self.layout.as_ref().map_or(1, |l| l.out_len)
    }
}

/// Streaming fold pass over `plan` to collect transform statistics (no dense output buffer).
///
/// # Errors
///
/// Propagates fold and geometry validation failures.
pub(crate) fn collect_transform_stats(
    mmap: &[u8],
    plan: &ReadPlan,
    op: &Operation,
    dtype: ElementDtype,
    policy: &FoldIoPolicy,
    tet_path: Option<&Path>,
) -> Result<(TransformStats, OperationPreviewFields, u64), TetError> {
    let Operation::Transform { method, axes } = op else {
        return Err(TetError::Validation(
            "internal: collect_transform_stats requires Operation::Transform".into(),
        ));
    };
    if !matches!(dtype, ElementDtype::F32 | ElementDtype::F64) {
        return Err(TetError::Validation(ERR_TRANSFORM_FLOAT.into()));
    }

    let layout = if axes.is_empty() {
        None
    } else {
        Some(partial_geometry::partial_axis_layout(plan, axes)?)
    };
    let (mut stats, mut fields, bytes) = collect_stats_for_method(&CollectStatsInput {
        mmap,
        plan,
        method: *method,
        dtype,
        policy,
        tet_path,
        layout: layout.as_ref(),
        axes,
    })?;
    if let Some(ref layout) = layout {
        stats.layout = Some(layout.clone());
        fields.reduced_shape = Some(layout.out_shape.clone());
    }
    stamp_preview_fields(&mut fields, *method, &stats, layout.as_ref());
    Ok((stats, fields, bytes))
}

fn stamp_preview_fields(
    fields: &mut OperationPreviewFields,
    method: TransformMethod,
    stats: &TransformStats,
    layout: Option<&PartialAxisLayout>,
) {
    fields.transform_method = Some(method.as_str().to_owned());
    if layout.is_some() {
        fields.reduced_mean = Some(stats.mean.clone());
        fields.reduced_std = Some(stats.std.clone());
        fields.reduced_min = Some(stats.min.clone());
        fields.reduced_max = Some(stats.max.clone());
        fields.reduced_norm_l1 = Some(stats.norm_l1.clone());
        fields.reduced_norm_l2 = Some(stats.norm_l2.clone());
    } else {
        assign_first_scalar(&mut fields.mean, &stats.mean);
        assign_first_scalar(&mut fields.std, &stats.std);
        assign_first_scalar(&mut fields.min, &stats.min);
        assign_first_scalar(&mut fields.max, &stats.max);
        assign_first_scalar(&mut fields.norm_l1, &stats.norm_l1);
        assign_first_scalar(&mut fields.norm_l2, &stats.norm_l2);
    }
}

fn assign_first_scalar(target: &mut Option<f64>, values: &[f64]) {
    if let Some(&v) = values.first() {
        *target = Some(v);
    }
}

/// Visit every finite logical element in `plan`, promoting `f32` to `f64` for the callback.
fn fold_transform_chunks<F>(
    mmap: &[u8],
    plan: &ReadPlan,
    dtype: ElementDtype,
    mut visit: F,
) -> Result<u64, TetError>
where
    F: FnMut(usize, f64) -> Result<(), TetError>,
{
    let mut total = 0u64;
    for c in &plan.chunks {
        let n = match dtype {
            ElementDtype::F32 => {
                visit_planned_chunk(mmap, plan, c, |li, v| visit(li, f64::from(v)))?
            }
            ElementDtype::F64 => visit_planned_chunk_f64(mmap, plan, c, &mut visit)?,
            _ => return Err(TetError::Validation(ERR_TRANSFORM_FLOAT.into())),
        };
        accumulate_chunk_read_bytes(&mut total, n)?;
    }
    Ok(total)
}

#[derive(Copy, Clone)]
struct CollectStatsInput<'a> {
    mmap: &'a [u8],
    plan: &'a ReadPlan,
    method: TransformMethod,
    dtype: ElementDtype,
    policy: &'a FoldIoPolicy,
    tet_path: Option<&'a Path>,
    layout: Option<&'a PartialAxisLayout>,
    axes: &'a [String],
}

fn fold_stat_vec(
    total_bytes: &mut u64,
    input: &CollectStatsInput<'_>,
    kind: ReductionKind,
    partial: bool,
) -> Result<Vec<f64>, TetError> {
    if partial {
        let folded = dispatch::partial_fold(
            input.mmap,
            input.plan,
            0,
            kind,
            input.axes,
            input.dtype,
            input.policy,
        )?;
        accumulate_chunk_read_bytes(total_bytes, folded.total_bytes_read_from_disk)?;
        Ok(extract_folded_values(&folded.operation, kind, true))
    } else {
        let folded = dispatch::scalar_fold(
            input.mmap,
            input.plan,
            0,
            kind,
            input.dtype,
            input.policy,
            input.tet_path,
        )?;
        accumulate_chunk_read_bytes(total_bytes, folded.total_bytes_read_from_disk)?;
        Ok(vec![
            extract_folded_values(&folded.operation, kind, false)
                .into_iter()
                .next()
                .unwrap_or(0.0),
        ])
    }
}

fn collect_stats_for_method(
    input: &CollectStatsInput<'_>,
) -> Result<(TransformStats, OperationPreviewFields, u64), TetError> {
    let CollectStatsInput {
        mmap,
        plan,
        method,
        dtype,
        layout,
        ..
    } = *input;
    let partial = layout.is_some();
    let mut stats = TransformStats {
        method,
        layout: layout.cloned(),
        mean: Vec::new(),
        std: Vec::new(),
        min: Vec::new(),
        max: Vec::new(),
        norm_l1: Vec::new(),
        norm_l2: Vec::new(),
        sum_exp: Vec::new(),
    };
    let mut total_bytes = 0u64;

    match method {
        TransformMethod::Center | TransformMethod::Zscore => {
            stats.mean = fold_stat_vec(&mut total_bytes, input, ReductionKind::Mean, partial)?;
            if method == TransformMethod::Zscore {
                stats.std = fold_stat_vec(&mut total_bytes, input, ReductionKind::Std, partial)?;
            }
        }
        TransformMethod::Scale => {
            stats.std = fold_stat_vec(&mut total_bytes, input, ReductionKind::Std, partial)?;
        }
        TransformMethod::Minmax => {
            stats.min = fold_stat_vec(&mut total_bytes, input, ReductionKind::Min, partial)?;
            stats.max = fold_stat_vec(&mut total_bytes, input, ReductionKind::Max, partial)?;
        }
        TransformMethod::L1 => {
            stats.norm_l1 = fold_stat_vec(&mut total_bytes, input, ReductionKind::NormL1, partial)?;
        }
        TransformMethod::L2 => {
            stats.norm_l2 = fold_stat_vec(&mut total_bytes, input, ReductionKind::NormL2, partial)?;
        }
        TransformMethod::Log1p | TransformMethod::Sqrt => {
            let ncells = layout.map_or(1, |l| l.out_len);
            let mut out_min = vec![f64::INFINITY; ncells];
            let shape = &plan.logical_selection_shape;
            let bytes = fold_transform_chunks(mmap, plan, dtype, |li, v| {
                if !v.is_finite() {
                    return Ok(());
                }
                let cell = partial_geometry::reduced_cell_index_or_global(li, shape, layout)?;
                out_min[cell] = out_min[cell].min(v);
                Ok(())
            })?;
            accumulate_chunk_read_bytes(&mut total_bytes, bytes)?;
            for m in &mut out_min {
                if !m.is_finite() {
                    *m = 0.0;
                }
            }
            stats.min = out_min;
        }
        TransformMethod::Softmax => {
            stats.max = fold_stat_vec(&mut total_bytes, input, ReductionKind::Max, partial)?;
            let ncells = stats.cell_len();
            let mut sum_exp = vec![0.0; ncells];
            let shape = &plan.logical_selection_shape;
            let max = stats.max.clone();
            let bytes = fold_transform_chunks(mmap, plan, dtype, |li, v| {
                if !v.is_finite() {
                    return Ok(());
                }
                let cell = partial_geometry::reduced_cell_index_or_global(li, shape, layout)?;
                sum_exp[cell] += (v - max[cell]).exp();
                Ok(())
            })?;
            accumulate_chunk_read_bytes(&mut total_bytes, bytes)?;
            stats.sum_exp = sum_exp;
        }
    }

    Ok((stats, OperationPreviewFields::default(), total_bytes))
}

/// Pull scalar or per-cell values from a completed fold into a `Vec` (one element when global).
fn extract_folded_values(
    fields: &OperationPreviewFields,
    kind: ReductionKind,
    partial: bool,
) -> Vec<f64> {
    if partial {
        match kind {
            ReductionKind::Mean => fields.reduced_mean.clone().unwrap_or_default(),
            ReductionKind::Std => fields.reduced_std.clone().unwrap_or_default(),
            ReductionKind::Min => fields.reduced_min.clone().unwrap_or_default(),
            ReductionKind::Max => fields.reduced_max.clone().unwrap_or_default(),
            ReductionKind::NormL1 => fields.reduced_norm_l1.clone().unwrap_or_default(),
            ReductionKind::NormL2 => fields.reduced_norm_l2.clone().unwrap_or_default(),
            _ => Vec::new(),
        }
    } else {
        let scalar = match kind {
            ReductionKind::Mean => fields.mean,
            ReductionKind::Std => fields.std,
            ReductionKind::Min => fields.min,
            ReductionKind::Max => fields.max,
            ReductionKind::NormL1 => fields.norm_l1,
            ReductionKind::NormL2 => fields.norm_l2,
            _ => None,
        };
        scalar.map_or_else(Vec::new, |v| vec![v])
    }
}
