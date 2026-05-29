//! Pass-1 fold stats for element-wise transforms.

use std::path::Path;

use crate::query::dispatch;
use crate::query::fold::{
    fold_policy::FoldIoPolicy,
    partial_geometry::{self, PartialAxisLayout},
    reduction::ReductionKind,
};
use crate::query::types::{Operation, OperationPreviewFields, ReadPlan, TetError};
use crate::utils::dtype::ElementDtype;

pub(crate) enum TransformStats {
    ZscoreScalar {
        mean: f64,
        std: f64,
    },
    ZscorePartial {
        mean: Vec<f64>,
        std: Vec<f64>,
        layout: PartialAxisLayout,
    },
    MinMaxScalar {
        min: f64,
        max: f64,
    },
    MinMaxPartial {
        min: Vec<f64>,
        max: Vec<f64>,
        layout: PartialAxisLayout,
    },
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
    if !matches!(dtype, ElementDtype::F32 | ElementDtype::F64) {
        return Err(TetError::Validation(
            "zscore and min_max_normalize require f32 or f64 datasets".into(),
        ));
    }
    let axes = op.axes();
    if axes.is_empty() {
        collect_scalar_stats(mmap, plan, op, dtype, policy, tet_path)
    } else {
        collect_partial_stats(mmap, plan, op, dtype, policy, axes)
    }
}

fn collect_scalar_stats(
    mmap: &[u8],
    plan: &ReadPlan,
    op: &Operation,
    dtype: ElementDtype,
    policy: &FoldIoPolicy,
    tet_path: Option<&Path>,
) -> Result<(TransformStats, OperationPreviewFields, u64), TetError> {
    match op {
        Operation::Zscore { .. } => {
            let mean_fold =
                dispatch::scalar_fold(mmap, plan, 0, ReductionKind::Mean, dtype, policy, tet_path)?;
            let mean = mean_fold.operation.mean.ok_or_else(|| {
                TetError::Validation("zscore pass-1 mean fold produced no result".into())
            })?;
            let std_fold =
                dispatch::scalar_fold(mmap, plan, 0, ReductionKind::Std, dtype, policy, tet_path)?;
            let std = std_fold.operation.std.ok_or_else(|| {
                TetError::Validation("zscore pass-1 std fold produced no result".into())
            })?;
            let bytes = mean_fold
                .total_bytes_read_from_disk
                .checked_add(std_fold.total_bytes_read_from_disk)
                .ok_or_else(|| TetError::Validation("total bytes read overflow".into()))?;
            let fields = OperationPreviewFields {
                mean: Some(mean),
                std: Some(std),
                ..Default::default()
            };
            Ok((TransformStats::ZscoreScalar { mean, std }, fields, bytes))
        }
        Operation::MinMaxNormalize { .. } => {
            let min_fold =
                dispatch::scalar_fold(mmap, plan, 0, ReductionKind::Min, dtype, policy, tet_path)?;
            let min = min_fold.operation.min.ok_or_else(|| {
                TetError::Validation("min_max_normalize pass-1 min fold produced no result".into())
            })?;
            let max_fold =
                dispatch::scalar_fold(mmap, plan, 0, ReductionKind::Max, dtype, policy, tet_path)?;
            let max = max_fold.operation.max.ok_or_else(|| {
                TetError::Validation("min_max_normalize pass-1 max fold produced no result".into())
            })?;
            let bytes = min_fold
                .total_bytes_read_from_disk
                .checked_add(max_fold.total_bytes_read_from_disk)
                .ok_or_else(|| TetError::Validation("total bytes read overflow".into()))?;
            let fields = OperationPreviewFields {
                min: Some(min),
                max: Some(max),
                ..Default::default()
            };
            Ok((TransformStats::MinMaxScalar { min, max }, fields, bytes))
        }
        _ => Err(TetError::Validation(
            "internal: collect_scalar_stats requires a transform operation".into(),
        )),
    }
}

fn collect_partial_stats(
    mmap: &[u8],
    plan: &ReadPlan,
    op: &Operation,
    dtype: ElementDtype,
    policy: &FoldIoPolicy,
    axes: &[String],
) -> Result<(TransformStats, OperationPreviewFields, u64), TetError> {
    let layout = partial_geometry::partial_axis_layout(plan, axes)?;
    match op {
        Operation::Zscore { .. } => {
            let mean_fold =
                dispatch::partial_fold(mmap, plan, 0, ReductionKind::Mean, axes, dtype, policy)?;
            let mean = mean_fold.operation.reduced_mean.ok_or_else(|| {
                TetError::Validation("zscore pass-1 partial mean fold produced no result".into())
            })?;
            let std_fold =
                dispatch::partial_fold(mmap, plan, 0, ReductionKind::Std, axes, dtype, policy)?;
            let std = std_fold.operation.reduced_std.ok_or_else(|| {
                TetError::Validation("zscore pass-1 partial std fold produced no result".into())
            })?;
            let bytes = mean_fold
                .total_bytes_read_from_disk
                .checked_add(std_fold.total_bytes_read_from_disk)
                .ok_or_else(|| TetError::Validation("total bytes read overflow".into()))?;
            let fields = OperationPreviewFields {
                reduced_shape: Some(layout.out_shape.clone()),
                reduced_mean: Some(mean.clone()),
                reduced_std: Some(std.clone()),
                ..Default::default()
            };
            Ok((
                TransformStats::ZscorePartial { mean, std, layout },
                fields,
                bytes,
            ))
        }
        Operation::MinMaxNormalize { .. } => {
            let min_fold =
                dispatch::partial_fold(mmap, plan, 0, ReductionKind::Min, axes, dtype, policy)?;
            let min = min_fold.operation.reduced_min.ok_or_else(|| {
                TetError::Validation(
                    "min_max_normalize pass-1 partial min fold produced no result".into(),
                )
            })?;
            let max_fold =
                dispatch::partial_fold(mmap, plan, 0, ReductionKind::Max, axes, dtype, policy)?;
            let max = max_fold.operation.reduced_max.ok_or_else(|| {
                TetError::Validation(
                    "min_max_normalize pass-1 partial max fold produced no result".into(),
                )
            })?;
            let bytes = min_fold
                .total_bytes_read_from_disk
                .checked_add(max_fold.total_bytes_read_from_disk)
                .ok_or_else(|| TetError::Validation("total bytes read overflow".into()))?;
            let fields = OperationPreviewFields {
                reduced_shape: Some(layout.out_shape.clone()),
                reduced_min: Some(min.clone()),
                reduced_max: Some(max.clone()),
                ..Default::default()
            };
            Ok((
                TransformStats::MinMaxPartial { min, max, layout },
                fields,
                bytes,
            ))
        }
        _ => Err(TetError::Validation(
            "internal: collect_partial_stats requires a transform operation".into(),
        )),
    }
}
