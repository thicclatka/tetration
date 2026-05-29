//! Pass-2 apply transforms while decoding the logical selection.

use std::path::Path;

use crate::query::decode::chunk_decode::{visit_planned_chunk, visit_planned_chunk_f64};
use crate::query::fold::partial_geometry;
use crate::query::materialize::shared::{
    check_materialized_nan_slice, scatter_fill_chunks, spill_byte_len_from_elem_count,
    spill_read_plan_pod_le_impl,
};
use crate::query::types::{ReadPlan, TetError};
use crate::utils::{f32_le, f64_le};

use super::stats::TransformStats;

fn stat_cell_index(li: usize, shape: &[u64], stats: &TransformStats) -> Result<usize, TetError> {
    match stats {
        TransformStats::ZscoreScalar { .. } | TransformStats::MinMaxScalar { .. } => Ok(0),
        TransformStats::ZscorePartial { layout, .. }
        | TransformStats::MinMaxPartial { layout, .. } => {
            let (oi, _) = partial_geometry::reduced_cell_index(li, shape, layout)?;
            Ok(oi)
        }
    }
}

fn apply_zscore(x: f64, mean: f64, std: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    if std == 0.0 || !std.is_finite() {
        return 0.0;
    }
    (x - mean) / std
}

fn apply_minmax(x: f64, min: f64, max: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    let range = max - min;
    if range == 0.0 || !range.is_finite() {
        return 0.0;
    }
    (x - min) / range
}

fn transform_f64(
    x: f64,
    li: usize,
    shape: &[u64],
    stats: &TransformStats,
) -> Result<f64, TetError> {
    let cell = stat_cell_index(li, shape, stats)?;
    Ok(match stats {
        TransformStats::ZscoreScalar { mean, std } => apply_zscore(x, *mean, *std),
        TransformStats::ZscorePartial { mean, std, .. } => apply_zscore(x, mean[cell], std[cell]),
        TransformStats::MinMaxScalar { min, max } => apply_minmax(x, *min, *max),
        TransformStats::MinMaxPartial { min, max, .. } => apply_minmax(x, min[cell], max[cell]),
    })
}

fn check_materialized_complete_f32(out: &[f32]) -> Result<(), TetError> {
    check_materialized_nan_slice(out, f32::is_nan)
}

fn check_materialized_complete_f64(out: &[f64]) -> Result<(), TetError> {
    check_materialized_nan_slice(out, f64::is_nan)
}

fn transform_scatter_fill_f32(
    mmap: &[u8],
    plan: &ReadPlan,
    out: &mut [f32],
    stats: &TransformStats,
) -> Result<u64, TetError> {
    let shape = plan.logical_selection_shape.clone();
    scatter_fill_chunks(mmap, plan, out, |mmap, plan, c, out| {
        visit_planned_chunk(mmap, plan, c, |li, v| {
            let y = transform_f64(f64::from(v), li, &shape, stats)? as f32;
            out[li] = y;
            Ok(())
        })
    })
}

fn transform_scatter_fill_f64(
    mmap: &[u8],
    plan: &ReadPlan,
    out: &mut [f64],
    stats: &TransformStats,
) -> Result<u64, TetError> {
    let shape = plan.logical_selection_shape.clone();
    scatter_fill_chunks(mmap, plan, out, |mmap, plan, c, out| {
        visit_planned_chunk_f64(mmap, plan, c, |li, v| {
            out[li] = transform_f64(v, li, &shape, stats)?;
            Ok(())
        })
    })
}

/// Materialize the transformed logical selection into a dense `f32` vector.
///
/// # Errors
///
/// Same validation failures as raw materialize, plus transform geometry errors.
pub(crate) fn transform_read_plan_f32_le_ram(
    mmap: &[u8],
    plan: &ReadPlan,
    stats: &TransformStats,
) -> Result<(Vec<f32>, u64), TetError> {
    let n = plan.logical_f32_element_count;
    let mut out = vec![f32::NAN; n];
    let bytes = transform_scatter_fill_f32(mmap, plan, &mut out, stats)?;
    check_materialized_complete_f32(&out)?;
    Ok((out, bytes))
}

/// Materialize the transformed logical selection into a dense `f64` vector.
///
/// # Errors
///
/// Same validation failures as raw materialize, plus transform geometry errors.
pub(crate) fn transform_read_plan_f64_le_ram(
    mmap: &[u8],
    plan: &ReadPlan,
    stats: &TransformStats,
) -> Result<(Vec<f64>, u64), TetError> {
    let n = plan.logical_f32_element_count;
    let mut out = vec![f64::NAN; n];
    let bytes = transform_scatter_fill_f64(mmap, plan, &mut out, stats)?;
    check_materialized_complete_f64(&out)?;
    Ok((out, bytes))
}

/// Spill the transformed logical selection as row-major little-endian POD to `path`.
///
/// # Errors
///
/// Propagates materialize and I/O failures.
pub(crate) fn transform_spill_f32_le(
    mmap: &[u8],
    plan: &ReadPlan,
    path: &Path,
    stats: &TransformStats,
) -> Result<u64, TetError> {
    let byte_len = spill_byte_len_from_elem_count(
        plan.logical_f32_element_count,
        f32_le::bytes_from_elem_count,
    )?;
    spill_read_plan_pod_le_impl(
        mmap,
        plan,
        path,
        byte_len,
        |mmap, plan, out| transform_scatter_fill_f32(mmap, plan, out, stats),
        check_materialized_complete_f32,
    )
}

/// Spill the transformed logical selection as row-major `f64` LE to `path`.
///
/// # Errors
///
/// Propagates materialize and I/O failures.
pub(crate) fn transform_spill_f64_le(
    mmap: &[u8],
    plan: &ReadPlan,
    path: &Path,
    stats: &TransformStats,
) -> Result<u64, TetError> {
    let byte_len = spill_byte_len_from_elem_count(
        plan.logical_f32_element_count,
        f64_le::bytes_from_elem_count,
    )?;
    spill_read_plan_pod_le_impl(
        mmap,
        plan,
        path,
        byte_len,
        |mmap, plan, out| transform_scatter_fill_f64(mmap, plan, out, stats),
        check_materialized_complete_f64,
    )
}
