//! Parallel chunk decode + scatter (Rayon).

use std::sync::atomic::{AtomicU64, Ordering};

use rayon::prelude::*;

use crate::query::types::{ReadPlan, TetError};

use super::chunk_decode::scatter_chunk_into_plan;
use super::materialize::{
    MaterializeReadPlanF32IntoOutcome, materialize_read_plan_f32_le_core,
    materialize_read_plan_f32_le_into_core, validate_read_plan_geometry,
};

pub(crate) fn materialize_scatter_fill_parallel(
    mmap: &[u8],
    plan: &ReadPlan,
    out: &mut [f32],
) -> Result<u64, TetError> {
    validate_read_plan_geometry(plan, out.len())?;
    if plan.chunks.is_empty() {
        return Ok(0);
    }
    let out_addr = out.as_mut_ptr() as usize;
    let out_len = out.len();
    let total_bytes = AtomicU64::new(0);
    plan.chunks.par_iter().try_for_each(|c| {
        // SAFETY: Each planned chunk covers disjoint global coordinates; logical row-major
        // indices written per chunk do not overlap.
        let out = unsafe { std::slice::from_raw_parts_mut(out_addr as *mut f32, out_len) };
        let n = scatter_chunk_into_plan(mmap, plan, c, out)?;
        total_bytes.fetch_add(n, Ordering::Relaxed);
        Ok::<(), TetError>(())
    })?;
    Ok(total_bytes.load(Ordering::Relaxed))
}

/// Like [`super::materialize::materialize_read_plan_f32_le`], but decodes planned chunks in parallel.
///
/// # Errors
///
/// Same failure modes as [`super::materialize::materialize_read_plan_f32_le`].
pub fn materialize_read_plan_f32_le_parallel(
    mmap: &[u8],
    plan: &ReadPlan,
    max_elements: Option<usize>,
) -> Result<(Vec<f32>, bool, u64), TetError> {
    materialize_read_plan_f32_le_core(mmap, plan, max_elements, materialize_scatter_fill_parallel)
}

/// Like [`super::materialize::materialize_read_plan_f32_le_into`], but decodes planned chunks in parallel.
///
/// # Errors
///
/// Same failure modes as [`super::materialize::materialize_read_plan_f32_le_into`].
pub fn materialize_read_plan_f32_le_into_parallel(
    mmap: &[u8],
    plan: &ReadPlan,
    max_elements: Option<usize>,
    dst: &mut [f32],
) -> Result<MaterializeReadPlanF32IntoOutcome, TetError> {
    materialize_read_plan_f32_le_into_core(
        mmap,
        plan,
        max_elements,
        dst,
        materialize_scatter_fill_parallel,
    )
}
