//! Pass-1 fold stats for element-wise transforms.

use std::path::Path;

use crate::query::decode::chunk_decode::{visit_planned_chunk, visit_planned_chunk_f64};
use crate::query::dispatch;
use crate::query::fold::{
    fold_policy::FoldIoPolicy,
    partial_geometry::{self, PartialAxisLayout},
    reduction::ReductionKind,
};
use crate::query::types::{Operation, OperationPreviewFields, ReadPlan, TetError};
use crate::utils::dtype::ElementDtype;

use super::TransformMethod;

/// Per reduced-cell statistics gathered in pass 1.
#[derive(Debug, Clone)]
pub(crate) struct TransformStats {
    pub method: TransformMethod,
    pub layout: Option<PartialAxisLayout>,
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
    pub min: Vec<f64>,
    pub max: Vec<f64>,
    pub norm_l1: Vec<f64>,
    pub norm_l2: Vec<f64>,
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
        return Err(TetError::Validation(
            "transform requires f32 or f64 datasets".into(),
        ));
    }
    if axes.is_empty() {
        collect_scalar_stats(mmap, plan, *method, dtype, policy, tet_path)
    } else {
        collect_partial_stats(mmap, plan, *method, dtype, policy, axes)
    }
}

fn collect_scalar_stats(
    mmap: &[u8],
    plan: &ReadPlan,
    method: TransformMethod,
    dtype: ElementDtype,
    policy: &FoldIoPolicy,
    tet_path: Option<&Path>,
) -> Result<(TransformStats, OperationPreviewFields, u64), TetError> {
    let (stats, mut fields, bytes) = collect_stats_for_method(&CollectStatsInput {
        mmap,
        plan,
        method,
        dtype,
        policy,
        tet_path,
        layout: None,
        axes: &[],
    })?;
    stamp_preview_fields(&mut fields, method, &stats, None);
    Ok((stats, fields, bytes))
}

fn collect_partial_stats(
    mmap: &[u8],
    plan: &ReadPlan,
    method: TransformMethod,
    dtype: ElementDtype,
    policy: &FoldIoPolicy,
    axes: &[String],
) -> Result<(TransformStats, OperationPreviewFields, u64), TetError> {
    let layout = partial_geometry::partial_axis_layout(plan, axes)?;
    let (mut stats, mut fields, bytes) = collect_stats_for_method(&CollectStatsInput {
        mmap,
        plan,
        method,
        dtype,
        policy,
        tet_path: None,
        layout: Some(&layout),
        axes,
    })?;
    stats.layout = Some(layout.clone());
    stamp_preview_fields(&mut fields, method, &stats, Some(&layout));
    fields.reduced_shape = Some(layout.out_shape);
    Ok((stats, fields, bytes))
}

fn stamp_preview_fields(
    fields: &mut OperationPreviewFields,
    method: TransformMethod,
    stats: &TransformStats,
    layout: Option<&PartialAxisLayout>,
) {
    fields.transform_method = Some(method.as_str().to_owned());
    let partial = layout.is_some();
    if partial {
        fields.reduced_mean = Some(stats.mean.clone());
        fields.reduced_std = Some(stats.std.clone());
        fields.reduced_min = Some(stats.min.clone());
        fields.reduced_max = Some(stats.max.clone());
        fields.reduced_norm_l1 = Some(stats.norm_l1.clone());
        fields.reduced_norm_l2 = Some(stats.norm_l2.clone());
    } else {
        if let Some(&m) = stats.mean.first() {
            fields.mean = Some(m);
        }
        if let Some(&s) = stats.std.first() {
            fields.std = Some(s);
        }
        if let Some(&mn) = stats.min.first() {
            fields.min = Some(mn);
        }
        if let Some(&mx) = stats.max.first() {
            fields.max = Some(mx);
        }
        if let Some(&n1) = stats.norm_l1.first() {
            fields.norm_l1 = Some(n1);
        }
        if let Some(&n2) = stats.norm_l2.first() {
            fields.norm_l2 = Some(n2);
        }
    }
}

fn scalar_fold_stat(
    total_bytes: &mut u64,
    mmap: &[u8],
    plan: &ReadPlan,
    kind: ReductionKind,
    dtype: ElementDtype,
    policy: &FoldIoPolicy,
    tet_path: Option<&Path>,
) -> Result<f64, TetError> {
    let folded = dispatch::scalar_fold(mmap, plan, 0, kind, dtype, policy, tet_path)?;
    *total_bytes = total_bytes
        .checked_add(folded.total_bytes_read_from_disk)
        .ok_or_else(|| TetError::Validation("total bytes read overflow".into()))?;
    Ok(extract_scalar(&folded.operation, kind))
}

fn partial_fold_stat(
    total_bytes: &mut u64,
    mmap: &[u8],
    plan: &ReadPlan,
    kind: ReductionKind,
    axes: &[String],
    dtype: ElementDtype,
    policy: &FoldIoPolicy,
) -> Result<Vec<f64>, TetError> {
    let folded = dispatch::partial_fold(mmap, plan, 0, kind, axes, dtype, policy)?;
    *total_bytes = total_bytes
        .checked_add(folded.total_bytes_read_from_disk)
        .ok_or_else(|| TetError::Validation("total bytes read overflow".into()))?;
    Ok(extract_partial(&folded.operation, kind))
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
        partial_fold_stat(
            total_bytes,
            input.mmap,
            input.plan,
            kind,
            input.axes,
            input.dtype,
            input.policy,
        )
    } else {
        Ok(vec![scalar_fold_stat(
            total_bytes,
            input.mmap,
            input.plan,
            kind,
            input.dtype,
            input.policy,
            input.tet_path,
        )?])
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
        policy: _,
        tet_path: _,
        layout,
        axes: _,
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
            stats.min = vec![f64::INFINITY; ncells];
            let bytes = fold_finite_min(mmap, plan, dtype, layout, &mut stats.min)?;
            total_bytes = total_bytes
                .checked_add(bytes)
                .ok_or_else(|| TetError::Validation("total bytes read overflow".into()))?;
        }
        TransformMethod::Softmax => {
            stats.max = fold_stat_vec(&mut total_bytes, input, ReductionKind::Max, partial)?;
            let ncells = stats.cell_len();
            stats.sum_exp = vec![0.0; ncells];
            let bytes =
                fold_softmax_sum_exp(mmap, plan, dtype, layout, &stats.max, &mut stats.sum_exp)?;
            total_bytes = total_bytes
                .checked_add(bytes)
                .ok_or_else(|| TetError::Validation("total bytes read overflow".into()))?;
        }
    }

    Ok((stats, OperationPreviewFields::default(), total_bytes))
}

fn extract_scalar(fields: &OperationPreviewFields, kind: ReductionKind) -> f64 {
    match kind {
        ReductionKind::Mean => fields.mean.unwrap_or(0.0),
        ReductionKind::Std => fields.std.unwrap_or(0.0),
        ReductionKind::Min => fields.min.unwrap_or(0.0),
        ReductionKind::Max => fields.max.unwrap_or(0.0),
        ReductionKind::NormL1 => fields.norm_l1.unwrap_or(0.0),
        ReductionKind::NormL2 => fields.norm_l2.unwrap_or(0.0),
        _ => 0.0,
    }
}

fn extract_partial(fields: &OperationPreviewFields, kind: ReductionKind) -> Vec<f64> {
    match kind {
        ReductionKind::Mean => fields.reduced_mean.clone().unwrap_or_default(),
        ReductionKind::Std => fields.reduced_std.clone().unwrap_or_default(),
        ReductionKind::Min => fields.reduced_min.clone().unwrap_or_default(),
        ReductionKind::Max => fields.reduced_max.clone().unwrap_or_default(),
        ReductionKind::NormL1 => fields.reduced_norm_l1.clone().unwrap_or_default(),
        ReductionKind::NormL2 => fields.reduced_norm_l2.clone().unwrap_or_default(),
        _ => Vec::new(),
    }
}

fn fold_finite_min(
    mmap: &[u8],
    plan: &ReadPlan,
    dtype: ElementDtype,
    layout: Option<&PartialAxisLayout>,
    out_min: &mut [f64],
) -> Result<u64, TetError> {
    let shape = &plan.logical_selection_shape;
    let visit = |li: usize, v: f64, mins: &mut [f64]| -> Result<(), TetError> {
        if !v.is_finite() {
            return Ok(());
        }
        let cell = if let Some(layout) = layout {
            let (oi, _) = partial_geometry::reduced_cell_index(li, shape, layout)?;
            oi
        } else {
            0
        };
        mins[cell] = mins[cell].min(v);
        Ok(())
    };

    let mut total = 0u64;
    for c in &plan.chunks {
        let n = match dtype {
            ElementDtype::F32 => {
                visit_planned_chunk(mmap, plan, c, |li, v| visit(li, f64::from(v), out_min))?
            }
            ElementDtype::F64 => {
                visit_planned_chunk_f64(mmap, plan, c, |li, v| visit(li, v, out_min))?
            }
            _ => {
                return Err(TetError::Validation(
                    "transform requires f32 or f64 datasets".into(),
                ));
            }
        };
        total = total
            .checked_add(n)
            .ok_or_else(|| TetError::Validation("total bytes read overflow".into()))?;
    }
    for m in out_min.iter_mut() {
        if !m.is_finite() {
            *m = 0.0;
        }
    }
    Ok(total)
}

fn fold_softmax_sum_exp(
    mmap: &[u8],
    plan: &ReadPlan,
    dtype: ElementDtype,
    layout: Option<&PartialAxisLayout>,
    max: &[f64],
    sum_exp: &mut [f64],
) -> Result<u64, TetError> {
    let shape = &plan.logical_selection_shape;
    let visit = |li: usize, v: f64, sums: &mut [f64]| -> Result<(), TetError> {
        if !v.is_finite() {
            return Ok(());
        }
        let cell = if let Some(layout) = layout {
            let (oi, _) = partial_geometry::reduced_cell_index(li, shape, layout)?;
            oi
        } else {
            0
        };
        let shifted = v - max[cell];
        sums[cell] += shifted.exp();
        Ok(())
    };

    let mut total = 0u64;
    for c in &plan.chunks {
        let n = match dtype {
            ElementDtype::F32 => {
                visit_planned_chunk(mmap, plan, c, |li, v| visit(li, f64::from(v), sum_exp))?
            }
            ElementDtype::F64 => {
                visit_planned_chunk_f64(mmap, plan, c, |li, v| visit(li, v, sum_exp))?
            }
            _ => {
                return Err(TetError::Validation(
                    "transform requires f32 or f64 datasets".into(),
                ));
            }
        };
        total = total
            .checked_add(n)
            .ok_or_else(|| TetError::Validation("total bytes read overflow".into()))?;
    }
    Ok(total)
}
