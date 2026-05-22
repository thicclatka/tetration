use serde::Serialize;

use super::document::Operation;

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
    /// `4 * product(shape)` when dtype is `f32` and the dataset matched.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset_f32_bytes: Option<u64>,
    /// Execution preferences from the `.tet` chunk index header.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_execution: Option<crate::catalog::FileExecutionSettingsV1>,
    /// Present when `matched` is false: names from the file’s dataset directory.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub available_datasets: Option<Vec<String>>,
}

/// Operation outputs merged into [`QueryExecutionPreview`] via [`From<OperationPreviewFields>`].
#[derive(Debug, Clone, Default)]
pub(crate) struct OperationPreviewFields {
    pub element_count: Option<usize>,
    pub sum: Option<f64>,
    pub mean: Option<f64>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub var: Option<f64>,
    pub std: Option<f64>,
    pub product: Option<f64>,
    pub norm_l1: Option<f64>,
    pub norm_l2: Option<f64>,
    pub all_finite: Option<bool>,
    pub any_nan: Option<bool>,
    pub reduced_shape: Option<Vec<u64>>,
    pub reduced_sum: Option<Vec<f64>>,
    pub reduced_mean: Option<Vec<f64>>,
    pub reduced_min: Option<Vec<f64>>,
    pub reduced_max: Option<Vec<f64>>,
    pub reduced_count: Option<Vec<f64>>,
    pub reduced_var: Option<Vec<f64>>,
    pub reduced_std: Option<Vec<f64>>,
    pub reduced_product: Option<Vec<f64>>,
    pub reduced_norm_l1: Option<Vec<f64>>,
    pub reduced_norm_l2: Option<Vec<f64>>,
    pub reduced_all_finite: Option<Vec<bool>>,
    pub reduced_any_nan: Option<Vec<bool>>,
}

/// First `f32` values read from planned chunk payloads (little-endian), capped for JSON safety.
/// When an [`Operation`] is present and execution runs, **`operation_*`** fields summarize the
/// **full** decoded logical tensor (row-major over the strided selection); see
/// `plan_query_with_tet_mmap`. Scalar reductions (`axes: []`) use `operation_sum`,
/// `operation_mean`, `operation_min`, `operation_max`, `operation_var`, `operation_std`, or
/// `operation_element_count` (for `count`), or `operation_product`. Partial reductions use `operation_reduced_shape` with matching
/// `operation_reduced_*` vectors (axes are decimal dimension indices).
#[derive(Debug, Clone, Default, Serialize)]
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_min: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_max: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_var: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_std: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_product: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_norm_l1: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_norm_l2: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_all_finite: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_any_nan: Option<bool>,
    /// How execution respected memory limits (`streaming_fold`, `capped_in_memory`, `mmap_spill`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_strategy: Option<&'static str>,
    /// When set, full logical selection was spilled as row-major `f32` LE to this path.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spill_f32_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spill_f32_bytes: Option<u64>,
    /// Resolved RAM budget used for this execution (bytes).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_budget_bytes: Option<u64>,
    /// Host available RAM when the budget was resolved, if detectable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub host_available_ram_bytes: Option<u64>,
    /// Logical selection size in bytes (`4 * logical_f32_element_count`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logical_selection_f32_bytes: Option<u64>,
    /// Shape after reducing along `operation.axes` (decimal dimension indices); row-major flattened payloads follow.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_reduced_shape: Option<Vec<u64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_reduced_sum: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_reduced_mean: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_reduced_min: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_reduced_max: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_reduced_count: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_reduced_var: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_reduced_std: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_reduced_product: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_reduced_norm_l1: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_reduced_norm_l2: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_reduced_all_finite: Option<Vec<bool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation_reduced_any_nan: Option<Vec<bool>>,
}

impl From<OperationPreviewFields> for QueryExecutionPreview {
    fn from(operation: OperationPreviewFields) -> Self {
        Self {
            operation_element_count: operation.element_count,
            operation_sum: operation.sum,
            operation_mean: operation.mean,
            operation_min: operation.min,
            operation_max: operation.max,
            operation_var: operation.var,
            operation_std: operation.std,
            operation_product: operation.product,
            operation_norm_l1: operation.norm_l1,
            operation_norm_l2: operation.norm_l2,
            operation_all_finite: operation.all_finite,
            operation_any_nan: operation.any_nan,
            operation_reduced_shape: operation.reduced_shape,
            operation_reduced_sum: operation.reduced_sum,
            operation_reduced_mean: operation.reduced_mean,
            operation_reduced_min: operation.reduced_min,
            operation_reduced_max: operation.reduced_max,
            operation_reduced_count: operation.reduced_count,
            operation_reduced_var: operation.reduced_var,
            operation_reduced_std: operation.reduced_std,
            operation_reduced_product: operation.reduced_product,
            operation_reduced_norm_l1: operation.reduced_norm_l1,
            operation_reduced_norm_l2: operation.reduced_norm_l2,
            operation_reduced_all_finite: operation.reduced_all_finite,
            operation_reduced_any_nan: operation.reduced_any_nan,
            ..Self::default()
        }
    }
}

impl QueryExecutionPreview {
    /// Decode preview only; all `operation_*` fields stay [`None`].
    #[must_use]
    pub fn decode_preview(
        total_bytes_read_from_disk: u64,
        f32_preview: Vec<f32>,
        f32_preview_truncated: bool,
    ) -> Self {
        Self {
            total_bytes_read_from_disk,
            f32_preview,
            f32_preview_truncated,
            ..Self::default()
        }
    }

    #[must_use]
    pub(crate) fn with_operation_and_io(
        total_bytes_read_from_disk: u64,
        f32_preview: Vec<f32>,
        f32_preview_truncated: bool,
        operation: OperationPreviewFields,
        memory_strategy: Option<&'static str>,
        spill_f32_path: Option<String>,
        spill_f32_bytes: Option<u64>,
    ) -> Self {
        Self {
            total_bytes_read_from_disk,
            f32_preview,
            f32_preview_truncated,
            memory_strategy,
            spill_f32_path,
            spill_f32_bytes,
            ..operation.into()
        }
    }
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
    pub read_plan: Option<super::plan::ReadPlan>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution: Option<QueryExecutionPreview>,
}
