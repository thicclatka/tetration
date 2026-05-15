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

/// Per-chunk file regions needed to satisfy the current plan (on-disk payload bytes; may be zstd-compressed).
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
    /// Dataset global shape (same rank as selection).
    pub dataset_shape: Vec<u64>,
    pub chunk_shape: Vec<u64>,
    /// Half-open global index box used for chunk intersection (`stop` exclusive).
    pub selection_box_start: Vec<u64>,
    pub selection_box_stop_exclusive: Vec<u64>,
    /// Per-axis step (≥ 1) for strided selection.
    pub selection_step: Vec<u64>,
    /// Extents of the strided selection grid (row-major order matches materialized `f32`).
    pub logical_selection_shape: Vec<u64>,
    /// Element count of the logical row-major buffer produced by [`crate::query::materialize_read_plan_f32_le`].
    pub logical_f32_element_count: usize,
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

/// First `f32` values read from planned chunk payloads (little-endian), capped for JSON safety.
/// When an [`Operation`] is present and execution runs, **`operation_*`** fields summarize the
/// **full** decoded logical tensor (row-major over the strided selection); see
/// `plan_query_with_tet_mmap`. Scalar reductions (`axes: []`) use `operation_sum` /
/// `operation_mean`. Partial reductions use `operation_reduced_shape` with
/// `operation_reduced_sum` or `operation_reduced_mean` (axes are decimal dimension indices).
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct QueryExecutionPreview {
    /// Sum of `stored_byte_len` for all planned chunks (bytes touched on disk).
    pub total_bytes_read_from_disk: u64,
    pub f32_preview: Vec<f32>,
    pub f32_preview_truncated: bool,
    /// Set when an operation ran: number of `f32` values aggregated (full planned decode).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_element_count: Option<usize>,
    /// Scalar result when `operation.axes` is empty (reduce all elements).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_sum: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_mean: Option<f64>,
    /// Shape after reducing along `operation.axes` (decimal dimension indices); row-major flattened payloads follow.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_reduced_shape: Option<Vec<u64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_reduced_sum: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_reduced_mean: Option<Vec<f64>>,
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
