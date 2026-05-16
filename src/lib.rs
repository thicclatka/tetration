//! **tetration** — Rust library for the Tetration mmap-oriented chunked tensor format.
//! The companion CLI binary is **`tet`** (see `src/bin/tet.rs`).

mod utils;

pub mod catalog;
pub mod layout;
pub mod query;

pub use catalog::{
    CHUNK_INDEX_HEADER_V1, CHUNK_PAYLOAD_CODEC_V1, CatalogError, ChunkIndexEntryV1,
    ChunkIndexHeaderV1, ChunkPayloadCodecV1, DTYPE_F32, DatasetRecordV1, MAX_NDIM,
    OneChunkRawWrite, RawArrayWrite, TetFileSummaryV1, chunk_coords_intersecting_global_box,
    chunk_coords_intersecting_strided, read_tet_summary_v1, validate_chunk_payloads,
    write_one_chunk_raw_file, write_raw_array_file,
};
pub use layout::{
    LAYOUT_VERSION_V1, LayoutError, LayoutOpenError, MAGIC, SUPERBLOCK_V1_LEN, SuperblockV1,
    create_empty_v1_file, mmap_file_read, open_superblock_v1, read_superblock_v1,
};
pub use query::{
    AxisSlice, CHUNK_TOUCH_POLICY, ChunkTouchPolicy, DatasetResolution,
    MaterializeReadPlanF32IntoOutcome, Operation, OutputHint, OutputHints, PlannedChunkIo,
    QueryDocument, QueryExecutionPreview, QueryResponse, ReadPlan, materialize_read_plan_f32_le,
    materialize_read_plan_f32_le_into, materialize_read_plan_f32_le_into_parallel,
    materialize_read_plan_f32_le_parallel, parse_query_json, plan_query_empty,
    plan_query_with_tet_mmap, planned_chunk_mmap_slices, validate_query,
};
#[doc(hidden)]
pub use utils::f32_le::{f32_count, read_f32_le_at, try_cast_f32_le};
