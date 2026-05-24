//! Scalar and partial-axis reductions over decoded selections.

pub mod parallel_fold;
pub mod partial_fold;
pub mod partial_geometry;
pub mod reduction;
pub mod shared;
pub(crate) mod variance_simd;

pub(crate) use shared::FoldPlanOutcome;
