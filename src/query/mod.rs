//! JSON query documents: validated plans, execution, and CLI formatters.
//!
//! Primary public surface for query/embedder use; see also [`crate::prelude`].

mod cli;
mod decode;
mod dispatch;
mod document;
mod document_wire;
pub(crate) mod engine;
mod execute;
pub(crate) mod fold;
pub(crate) mod materialize;
mod plan;
mod resolve_axes;
mod resolve_selection;
pub(crate) mod types;

pub use crate::catalog::DEFAULT_MEMORY_BUDGET_PERCENT_BPS;
pub use cli::{
    CliQueryHistoryEntry, DEFAULT_INFO_CHUNK_TABLE_LIMIT, HistoryExecuteFilter, HistoryListFilter,
    HistorySettings, InfoListFilter, InfoMetadataDisplay, InfoViewSections, QueryOutputFormat,
    append_cli_query_history, clear_cli_query_history, cli_query_history_enabled,
    cli_query_history_max, cli_query_history_path, format_history_list_json,
    format_history_list_text, format_info_json, format_info_quiet, format_info_text,
    format_query_response, format_query_stderr_hints, get_cli_query_history_entry,
    history_entry_mode, info_view_sections_from_flags, list_cli_query_history,
    parse_history_execute_filter,
};
pub use document::{QueryLimits, parse_query_json, validate_query};
#[doc(hidden)]
pub use engine::TempSpillFile;
pub use engine::{
    DEFAULT_MEMORY_BUDGET_BYTES, ExecutionBudget, MaterializeReadPlanF32IntoOutcome,
    MemoryStrategy, SpillPathAllowlist, materialize_read_plan_f32_le,
    materialize_read_plan_f32_le_into, materialize_read_plan_f32_le_into_parallel,
    materialize_read_plan_f32_le_parallel, materialize_read_plan_f64_le,
    materialize_read_plan_i16_le, materialize_read_plan_i32_le, materialize_read_plan_i64_le,
    materialize_read_plan_u8_le, materialize_read_plan_u16_le, plan_query_empty,
    plan_query_with_tet_mmap, plan_query_with_tet_mmap_ex, planned_chunk_mmap_slices,
    spill_read_plan_f32_le, spill_read_plan_i16_le, spill_read_plan_i32_le, spill_read_plan_i64_le,
    spill_read_plan_u8_le, spill_read_plan_u16_le,
};
pub use execute::{ExecuteQueryOptions, execute_query_document, execute_query_json};
pub use types::{
    AxisSlice, CHUNK_TOUCH_POLICY, ChunkTouchPolicy, DatasetResolution, ExecutionHints, Operation,
    OutputHint, OutputHints, PlannedChunkIo, QueryDocument, QueryExecutionPreview, QueryResponse,
    ReadPlan, TetError,
};
