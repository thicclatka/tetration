//! Pass-2 apply transforms while decoding the logical selection.

use std::cell::RefCell;
use std::path::Path;

use crate::query::decode::chunk_decode::{visit_planned_chunk, visit_planned_chunk_f64};
use crate::query::fold::partial_geometry;
use crate::query::materialize::shared::{
    scatter_fill_chunks, spill_byte_len_from_elem_count, spill_read_plan_pod_le_impl,
};
use crate::query::types::{ReadPlan, TetError, TransformMethod};
use crate::utils::{f32_le, f64_le};

use super::stats::TransformStats;
use super::warnings::TransformWarnings;

fn stat_cell_index(li: usize, shape: &[u64], stats: &TransformStats) -> Result<usize, TetError> {
    if let Some(layout) = &stats.layout {
        let (oi, _) = partial_geometry::reduced_cell_index(li, shape, layout)?;
        Ok(oi)
    } else {
        Ok(0)
    }
}

fn div_or_nan(numer: f64, denom: f64, li: usize, warnings: &mut TransformWarnings) -> f64 {
    if denom == 0.0 || !denom.is_finite() {
        warnings.record_div_by_zero(li as u64);
        return f64::NAN;
    }
    numer / denom
}

fn transform_f64(
    x: f64,
    li: usize,
    shape: &[u64],
    stats: &TransformStats,
    warnings: &mut TransformWarnings,
) -> Result<f64, TetError> {
    if !x.is_finite() {
        return Ok(x);
    }
    let cell = stat_cell_index(li, shape, stats)?;
    Ok(match stats.method {
        TransformMethod::Center => x - stats.mean[cell],
        TransformMethod::Zscore => {
            let mean = stats.mean[cell];
            let std = stats.std[cell];
            div_or_nan(x - mean, std, li, warnings)
        }
        TransformMethod::Scale => div_or_nan(x, stats.std[cell], li, warnings),
        TransformMethod::Minmax => {
            let min = stats.min[cell];
            let range = stats.max[cell] - min;
            div_or_nan(x - min, range, li, warnings)
        }
        TransformMethod::L1 => div_or_nan(x, stats.norm_l1[cell], li, warnings),
        TransformMethod::L2 => div_or_nan(x, stats.norm_l2[cell], li, warnings),
        TransformMethod::Log1p => (x - stats.min[cell]).ln_1p(),
        TransformMethod::Sqrt => {
            let shifted = x - stats.min[cell];
            if shifted < 0.0 {
                warnings.record_div_by_zero(li as u64);
                f64::NAN
            } else {
                shifted.sqrt()
            }
        }
        TransformMethod::Softmax => {
            let max = stats.max[cell];
            let sum_exp = stats.sum_exp[cell];
            if sum_exp == 0.0 || !sum_exp.is_finite() {
                warnings.record_div_by_zero(li as u64);
                f64::NAN
            } else {
                (x - max).exp() / sum_exp
            }
        }
    })
}

fn transform_scatter_fill_f32(
    mmap: &[u8],
    plan: &ReadPlan,
    out: &mut [f32],
    stats: &TransformStats,
    warnings: &RefCell<TransformWarnings>,
) -> Result<u64, TetError> {
    let shape = plan.logical_selection_shape.clone();
    scatter_fill_chunks(mmap, plan, out, |mmap, plan, c, out| {
        visit_planned_chunk(mmap, plan, c, |li, v| {
            let y =
                transform_f64(f64::from(v), li, &shape, stats, &mut warnings.borrow_mut())? as f32;
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
    warnings: &RefCell<TransformWarnings>,
) -> Result<u64, TetError> {
    let shape = plan.logical_selection_shape.clone();
    scatter_fill_chunks(mmap, plan, out, |mmap, plan, c, out| {
        visit_planned_chunk_f64(mmap, plan, c, |li, v| {
            out[li] = transform_f64(v, li, &shape, stats, &mut warnings.borrow_mut())?;
            Ok(())
        })
    })
}

/// Materialize the transformed logical selection into a dense `f32` vector.
///
/// # Errors
///
/// Propagates materialize and geometry failures.
pub(crate) fn transform_read_plan_f32_le_ram(
    mmap: &[u8],
    plan: &ReadPlan,
    stats: &TransformStats,
    warnings: &mut TransformWarnings,
) -> Result<(Vec<f32>, u64), TetError> {
    let cell = RefCell::new(std::mem::take(warnings));
    let n = plan.logical_f32_element_count;
    let mut out = vec![f32::NAN; n];
    let bytes = transform_scatter_fill_f32(mmap, plan, &mut out, stats, &cell)?;
    *warnings = cell.into_inner();
    Ok((out, bytes))
}

/// Materialize the transformed logical selection into a dense `f64` vector.
///
/// # Errors
///
/// Propagates materialize and geometry failures.
pub(crate) fn transform_read_plan_f64_le_ram(
    mmap: &[u8],
    plan: &ReadPlan,
    stats: &TransformStats,
    warnings: &mut TransformWarnings,
) -> Result<(Vec<f64>, u64), TetError> {
    let cell = RefCell::new(std::mem::take(warnings));
    let n = plan.logical_f32_element_count;
    let mut out = vec![f64::NAN; n];
    let bytes = transform_scatter_fill_f64(mmap, plan, &mut out, stats, &cell)?;
    *warnings = cell.into_inner();
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
    warnings: &mut TransformWarnings,
) -> Result<u64, TetError> {
    let cell = RefCell::new(std::mem::take(warnings));
    let byte_len = spill_byte_len_from_elem_count(
        plan.logical_f32_element_count,
        f32_le::bytes_from_elem_count,
    )?;
    let bytes = spill_read_plan_pod_le_impl(
        mmap,
        plan,
        path,
        byte_len,
        |mmap, plan, out| transform_scatter_fill_f32(mmap, plan, out, stats, &cell),
        None,
    )?;
    *warnings = cell.into_inner();
    Ok(bytes)
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
    warnings: &mut TransformWarnings,
) -> Result<u64, TetError> {
    let cell = RefCell::new(std::mem::take(warnings));
    let byte_len = spill_byte_len_from_elem_count(
        plan.logical_f32_element_count,
        f64_le::bytes_from_elem_count,
    )?;
    let bytes = spill_read_plan_pod_le_impl(
        mmap,
        plan,
        path,
        byte_len,
        |mmap, plan, out| transform_scatter_fill_f64(mmap, plan, out, stats, &cell),
        None,
    )?;
    *warnings = cell.into_inner();
    Ok(bytes)
}
