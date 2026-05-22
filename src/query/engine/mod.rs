//! Query engine: mmap-backed read planning, `f32` materialization, and execution preview.

mod budget;
mod chunk_decode;
mod fold;
mod indexing;
mod materialize;
mod operations;
mod parallel;
mod partial_fold;
mod read_plan;
mod reduction;
mod run;
mod selection;

pub use budget::{DEFAULT_MEMORY_BUDGET_BYTES, ExecutionBudget, MemoryStrategy};
pub use chunk_decode::planned_chunk_mmap_slices;
pub use materialize::{
    MaterializeReadPlanF32IntoOutcome, materialize_read_plan_f32_le,
    materialize_read_plan_f32_le_into, spill_read_plan_f32_le,
};
pub use parallel::{
    materialize_read_plan_f32_le_into_parallel, materialize_read_plan_f32_le_parallel,
};
pub use run::{plan_query_empty, plan_query_with_tet_mmap};
