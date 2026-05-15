//! JSON query / response types and errors.

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TetError {
    #[error("invalid query JSON: {0}")]
    InvalidJson(#[from] serde_json::Error),
    #[error("validation: {0}")]
    Validation(String),
    #[error(transparent)]
    Catalog(#[from] crate::catalog::CatalogError),
}

/// Per-axis slice: `start` inclusive, `stop` exclusive, `step` ≥ 1 when present.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AxisSlice {
    pub start: Option<u64>,
    pub stop: Option<u64>,
    pub step: Option<u64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Operation {
    Sum { axes: Vec<String> },
    Mean { axes: Vec<String> },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputHint {
    InlineJson,
    SpillArray { handle: String },
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct OutputHints {
    #[serde(default)]
    pub preferred: Option<OutputHint>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct QueryDocument {
    #[serde(default)]
    pub layout_version: Option<u32>,
    pub dataset: String,
    #[serde(default)]
    pub selection: Option<Vec<AxisSlice>>,
    #[serde(default)]
    pub operation: Option<Operation>,
    #[serde(default)]
    pub output: Option<OutputHints>,
}

/// Per-chunk file regions needed to satisfy the current plan (raw payload bytes).
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct PlannedChunkIo {
    pub chunk_index: Vec<u64>,
    pub payload_offset: u64,
    pub stored_byte_len: u64,
    pub raw_byte_len: u64,
    pub codec: u32,
}

/// Chunk-level read plan: which tiles touch the resolved global selection box.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct ReadPlan {
    /// How `selection` maps to chunk coordinates in this build.
    pub chunk_touch_policy: &'static str,
    pub chunk_count: usize,
    pub total_stored_bytes: u64,
    pub chunks: Vec<PlannedChunkIo>,
}

/// Stable string tokens for [`ReadPlan::chunk_touch_policy`] (JSON wire compatibility).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkTouchPolicy {
    /// Half-open intervals with effective step 1 on every axis (JSON `step` omitted or 1).
    pub dense_half_open_unit_step: &'static str,
    /// Per-axis JSON `step` applied when deciding which chunks are touched.
    pub strided_half_open: &'static str,
}

/// Singleton tokens for chunk-touch policy strings on the wire.
pub const CHUNK_TOUCH_POLICY: ChunkTouchPolicy = ChunkTouchPolicy {
    dense_half_open_unit_step: "dense_half_open_intervals_per_axis_json_step_ignored_for_chunk_touch_list",
    strided_half_open: "strided_half_open_per_axis_json_step_applied_to_chunk_touch_list",
};

/// Result of matching `QueryDocument::dataset` against a mmap’d `.tet` catalog.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct DatasetResolution {
    pub matched: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset_index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dtype: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shape: Option<Vec<u64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_shape: Option<Vec<u64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_index_rows: Option<usize>,
    /// Present when `matched` is false: names from the file’s dataset directory.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub available_datasets: Option<Vec<String>>,
}

/// First `f32` values read from planned raw chunk payloads (little-endian), capped for JSON safety.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct QueryExecutionPreview {
    /// Sum of `stored_byte_len` for all planned chunks (bytes touched on disk).
    pub total_bytes_read_from_disk: u64,
    pub f32_preview: Vec<f32>,
    pub f32_preview_truncated: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct QueryResponse {
    pub status: &'static str,
    pub accepted: bool,
    pub layout_version: Option<u32>,
    pub dataset: String,
    pub selection_axes: Option<usize>,
    pub operation: Option<Operation>,
    pub message: String,
    /// When planning with `--tet`, the path echoed back for logs and scripts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tet_file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub catalog: Option<DatasetResolution>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub read_plan: Option<ReadPlan>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution: Option<QueryExecutionPreview>,
}
