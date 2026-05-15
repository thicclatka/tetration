//! JSON query documents: validated plans for reads and basic operations.

mod document;
mod plan;
mod types;

pub use document::{parse_query_json, validate_query};
pub use plan::{
    materialize_read_plan_f32_le, plan_query, plan_query_with_tet_mmap, planned_chunk_mmap_slices,
};
pub use types::{
    AxisSlice, CHUNK_TOUCH_POLICY, ChunkTouchPolicy, DatasetResolution, Operation, OutputHint,
    OutputHints, PlannedChunkIo, QueryDocument, QueryExecutionPreview, QueryResponse, ReadPlan,
    TetError,
};
