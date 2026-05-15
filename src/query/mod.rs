//! JSON query documents: validated plans for reads and basic operations.

mod document;
mod engine;
mod types;

pub use document::{parse_query_json, validate_query};
pub use engine::{
    MaterializeReadPlanF32IntoOutcome, materialize_read_plan_f32_le,
    materialize_read_plan_f32_le_into, materialize_read_plan_f32_le_into_parallel,
    materialize_read_plan_f32_le_parallel, plan_query_empty, plan_query_with_tet_mmap,
    planned_chunk_mmap_slices,
};
pub use types::{
    AxisSlice, CHUNK_TOUCH_POLICY, ChunkTouchPolicy, DatasetResolution, Operation, OutputHint,
    OutputHints, PlannedChunkIo, QueryDocument, QueryExecutionPreview, QueryResponse, ReadPlan,
    TetError,
};
