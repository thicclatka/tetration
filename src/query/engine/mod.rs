//! Query engine: mmap-backed read planning, `f32` materialization, and execution preview.

mod indexing;
mod materialize;
mod operations;
mod parallel;
mod read_plan;
mod run;
mod selection;

pub use materialize::{
    MaterializeReadPlanF32IntoOutcome, materialize_read_plan_f32_le,
    materialize_read_plan_f32_le_into, planned_chunk_mmap_slices,
};
pub use parallel::{
    materialize_read_plan_f32_le_into_parallel, materialize_read_plan_f32_le_parallel,
};
pub use run::{plan_query_empty, plan_query_with_tet_mmap};
