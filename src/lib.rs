//! **tetration** — Rust library for the Tetration mmap-oriented chunked tensor format.
//! The companion CLI binary is **`tet`** (see `src/bin/tet.rs` and `src/bin/tet/`).

mod utils;

pub mod catalog;
pub mod convert;
pub mod layout;
pub mod query;

pub use catalog::{
    CHUNK_INDEX_HEADER_V1, CHUNK_PAYLOAD_CODEC_V1, CatalogError, ChunkIndexEntryV1,
    ChunkIndexHeaderV1, ChunkPayloadCodecV1, DATASET_DTYPE_TAG_V1,
    DEFAULT_MEMORY_BUDGET_PERCENT_BPS, DatasetDtypeTagV1, DatasetRecordV1, FileExecutionSettingsV1,
    HistoryEventV1, MAX_NDIM, OneChunkRawWrite, RawArrayWrite, TetFileSummaryV1,
    chunk_coords_intersecting_global_box, chunk_coords_intersecting_strided,
    f32_tensor_bytes_from_shape, f64_tensor_bytes_from_shape, i32_tensor_bytes_from_shape,
    i64_tensor_bytes_from_shape, read_tet_summary_v1, validate_chunk_payloads,
    write_multi_raw_array_file, write_one_chunk_raw_file, write_raw_array_file,
};
pub use convert::{
    ConvertCompressionSuffixes, ConvertDatasetSummary, ConvertError, ConvertInputFormat,
    ConvertProgress, ConvertReport, Hdf5ConvertInput, NetcdfConvertInput, ZarrConvertInput,
    convert_to_tet, convert_to_tet_with_progress, convert_zarr_to_tet,
    convert_zarr_to_tet_with_progress, default_parallel_jobs, detect_convert_format,
    is_zarr_v3_directory, resolve_parallel_jobs,
};
#[cfg(feature = "tetration-hdf5")]
pub use convert::{convert_h5_to_tet, convert_h5_to_tet_with_progress};
#[cfg(feature = "tetration-netcdf")]
pub use convert::{convert_netcdf_to_tet, convert_netcdf_to_tet_with_progress};
pub use layout::{
    LAYOUT_VERSION_V1, LayoutError, LayoutOpenError, MAGIC, SUPERBLOCK_FLAG_HISTORY_FOOTER,
    SUPERBLOCK_V1_LEN, SuperblockV1, create_empty_v1_file, mmap_file_read, open_superblock_v1,
    read_superblock_v1,
};
#[doc(hidden)]
pub use query::TempSpillFile;
pub use query::{
    AxisSlice, CHUNK_TOUCH_POLICY, ChunkTouchPolicy, CliQueryHistoryEntry,
    DEFAULT_MEMORY_BUDGET_BYTES, DatasetResolution, ExecutionBudget, ExecutionHints,
    HistoryExecuteFilter, HistoryListFilter, HistorySettings, MaterializeReadPlanF32IntoOutcome,
    MemoryStrategy, Operation, OutputHint, OutputHints, PlannedChunkIo, QueryDocument,
    QueryExecutionPreview, QueryLimits, QueryOutputFormat, QueryResponse, ReadPlan,
    SpillPathAllowlist, append_cli_query_history, clear_cli_query_history, cli_query_history_max,
    cli_query_history_path, format_history_list_json, format_history_list_text,
    format_query_response, format_query_stderr_hints, get_cli_query_history_entry,
    history_entry_mode, list_cli_query_history, materialize_read_plan_f32_le,
    materialize_read_plan_f32_le_into, materialize_read_plan_f32_le_into_parallel,
    materialize_read_plan_f32_le_parallel, materialize_read_plan_f64_le,
    materialize_read_plan_i32_le, materialize_read_plan_i64_le, parse_history_execute_filter,
    parse_query_json, plan_query_empty, plan_query_with_tet_mmap, plan_query_with_tet_mmap_ex,
    planned_chunk_mmap_slices, spill_read_plan_f32_le, spill_read_plan_i32_le,
    spill_read_plan_i64_le, validate_query,
};
#[doc(hidden)]
pub use utils::f32_le::{f32_count, read_f32_le_at, try_cast_f32_le};
#[doc(hidden)]
pub use utils::host_memory::available_memory_bytes;
