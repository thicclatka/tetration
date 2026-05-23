//! JSON query documents: validated plans for reads and basic operations.

mod document;
mod engine;
mod types;

pub use crate::catalog::DEFAULT_MEMORY_BUDGET_PERCENT_BPS;
pub use document::{QueryLimits, parse_query_json, validate_query};
#[doc(hidden)]
pub use engine::TempSpillFile;
pub use engine::{
    DEFAULT_MEMORY_BUDGET_BYTES, ExecutionBudget, MaterializeReadPlanF32IntoOutcome,
    MemoryStrategy, SpillPathAllowlist, materialize_read_plan_f32_le,
    materialize_read_plan_f32_le_into, materialize_read_plan_f32_le_into_parallel,
    materialize_read_plan_f32_le_parallel, plan_query_empty, plan_query_with_tet_mmap,
    plan_query_with_tet_mmap_ex, planned_chunk_mmap_slices, spill_read_plan_f32_le,
};
pub use types::{
    AxisSlice, CHUNK_TOUCH_POLICY, ChunkTouchPolicy, DatasetResolution, ExecutionHints, Operation,
    OutputHint, OutputHints, PlannedChunkIo, QueryDocument, QueryExecutionPreview, QueryResponse,
    ReadPlan, TetError,
};
