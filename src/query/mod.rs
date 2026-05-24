//! JSON query documents: validated plans for reads and basic operations.

mod cli;
mod decode;
mod dispatch;
mod document;
mod document_wire;
mod engine;
mod fold;
mod materialize;
mod plan;
mod types;

pub use crate::catalog::DEFAULT_MEMORY_BUDGET_PERCENT_BPS;
pub use cli::{
    CliQueryHistoryEntry, HistoryExecuteFilter, HistoryListFilter, HistorySettings,
    QueryOutputFormat, append_cli_query_history, clear_cli_query_history,
    cli_query_history_enabled, cli_query_history_max, cli_query_history_path,
    format_history_list_json, format_history_list_text, format_query_response,
    format_query_stderr_hints, get_cli_query_history_entry, history_entry_mode,
    list_cli_query_history, parse_history_execute_filter,
};
pub use document::{QueryLimits, parse_query_json, validate_query};
#[doc(hidden)]
pub use engine::TempSpillFile;
pub use engine::{
    DEFAULT_MEMORY_BUDGET_BYTES, ExecutionBudget, MaterializeReadPlanF32IntoOutcome,
    MemoryStrategy, SpillPathAllowlist, materialize_read_plan_f32_le,
    materialize_read_plan_f32_le_into, materialize_read_plan_f32_le_into_parallel,
    materialize_read_plan_f32_le_parallel, materialize_read_plan_f64_le,
    materialize_read_plan_i32_le, materialize_read_plan_i64_le, plan_query_empty,
    plan_query_with_tet_mmap, plan_query_with_tet_mmap_ex, planned_chunk_mmap_slices,
    spill_read_plan_f32_le, spill_read_plan_i32_le, spill_read_plan_i64_le,
};
pub use types::{
    AxisSlice, CHUNK_TOUCH_POLICY, ChunkTouchPolicy, DatasetResolution, ExecutionHints, Operation,
    OutputHint, OutputHints, PlannedChunkIo, QueryDocument, QueryExecutionPreview, QueryResponse,
    ReadPlan, TetError,
};
