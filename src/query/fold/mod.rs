//! Scalar and partial-axis reductions over decoded selections.

pub mod partial_fold;
pub mod partial_geometry;
pub mod reduction;
pub mod shared;

pub(crate) use shared::FoldPlanOutcome;
