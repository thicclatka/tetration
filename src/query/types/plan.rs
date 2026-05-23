use serde::Serialize;

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
