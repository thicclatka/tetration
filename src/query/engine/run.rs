use std::path::{Path, PathBuf};

use crate::catalog::{
    CatalogError, DTYPE_F32, TetFileSummaryV1, chunk_coords_intersecting_strided,
    f32_tensor_bytes_from_shape, read_tet_summary_v1,
};
use crate::query::types::{
    CHUNK_TOUCH_POLICY, DatasetResolution, QueryDocument, QueryExecutionPreview, QueryResponse,
    ReadPlan, TetError,
};

use super::budget::ExecutionBudget;
use super::operations::build_execution_preview;
use super::read_plan::{ReadPlanGeometry, build_read_plan};
use super::selection::resolved_dense_global_box;
use super::spill_policy::SpillPathAllowlist;

fn map_geometry_catalog_error(e: CatalogError) -> TetError {
    match e {
        CatalogError::InvalidWriteSpec(msg) => TetError::Validation(format!(
            "selection does not form a valid global box for this dataset: {msg}"
        )),
        other => TetError::Catalog(other),
    }
}

fn query_response(
    doc: &QueryDocument,
    message: impl Into<String>,
    tet_path: Option<&str>,
    catalog: Option<DatasetResolution>,
    read_plan: Option<ReadPlan>,
    execution: Option<QueryExecutionPreview>,
) -> QueryResponse {
    QueryResponse {
        status: "planned",
        accepted: true,
        layout_version: doc.layout_version,
        dataset: doc.dataset.clone(),
        selection_axes: doc.selection.as_ref().map(Vec::len),
        operation: doc.operation.clone(),
        message: message.into(),
        tet_file: tet_path.map(str::to_string),
        catalog,
        read_plan,
        execution,
    }
}

fn base_response(doc: &QueryDocument, message: impl Into<String>) -> QueryResponse {
    query_response(doc, message, None, None, None, None)
}

/// Build a response echoing the plan without consulting a `.tet` file (no mmap).
#[must_use]
pub fn plan_query_empty(doc: &QueryDocument) -> QueryResponse {
    base_response(
        doc,
        "query accepted and validated; pass `--tet PATH` for catalog + read_plan, or add `--execute` with `--tet` for a capped raw f32 mmap preview",
    )
}

/// Like [`plan_query_empty`], but read catalog metadata from `mmap` (full `.tet` bytes).
///
/// `tet_path` is echoed in the JSON as `tet_file` when `Some`.
///
/// When the dataset matches, [`QueryResponse::read_plan`] lists chunk payloads that intersect the
/// per-axis selection (default full tensor). Non-unit JSON `step` uses strided chunk selection
/// (see [`crate::query::CHUNK_TOUCH_POLICY`]).
///
/// `raw_f32_preview_max`: when `Some(n)` with `n > 0`, read planned chunk payloads from `mmap`
/// (raw or zstd `f32` tiles, codecs **0** / **1**) and attach up to `n` decoded little-endian `f32`
/// values under [`QueryResponse::execution`] (requires dataset
/// [`DTYPE_F32`](crate::catalog::DTYPE_F32)). When `Some(0)`, only attach execution when
/// [`QueryDocument::operation`] is set: the full logical tensor is still decoded for aggregation,
/// but `f32_preview` is empty. When [`QueryDocument::operation`] is set, a limit must be passed
/// (use `0` to skip preview floats). Partial reductions populate `operation_reduced_*` fields;
/// scalar reductions (`sum`, `mean`, `min`, `max`, `count`, `var`, `std` with `axes: []`) use single-pass fold fields such as `operation_sum` / `operation_var`.
///
/// # Errors
///
/// Returns [`TetError::Catalog`] when the file is not a valid v1 catalog view, or
/// [`TetError::Validation`] when `selection` is inconsistent with dataset rank or shape, when an
/// `operation` is set but `raw_f32_preview_max` is `None`, when `Some(0)` is used without an
/// `operation`, when preview or aggregation requires a non-`f32` dataset, or when payload bytes
/// cannot be read or decoded from `mmap`.
pub fn plan_query_with_tet_mmap(
    doc: &QueryDocument,
    tet_path: Option<&str>,
    mmap: &[u8],
    raw_f32_preview_max: Option<usize>,
) -> Result<QueryResponse, TetError> {
    plan_query_with_tet_mmap_ex(doc, tet_path, mmap, raw_f32_preview_max, None)
}

/// Like [`plan_query_with_tet_mmap`], with an optional spill path allowlist.
///
/// When `spill_allowlist` is `None` and `tet_path` is set, [`SpillPathAllowlist::default_for_tet`]
/// applies (`.tet` parent + platform cache dirs). Pass an explicit allowlist to override defaults
/// entirely, or build [`SpillPathAllowlist::default_for_tet`] with extra `--spill-allow` roots.
///
/// # Errors
///
/// Same as [`plan_query_with_tet_mmap`], plus [`TetError::Validation`] when the default spill
/// allowlist cannot be built from `tet_path`, when export spill or materialize-required operations
/// need an allowlist but none is available, or when a spill path fails allowlist validation.
pub fn plan_query_with_tet_mmap_ex(
    doc: &QueryDocument,
    tet_path: Option<&str>,
    mmap: &[u8],
    raw_f32_preview_max: Option<usize>,
    spill_allowlist: Option<&SpillPathAllowlist>,
) -> Result<QueryResponse, TetError> {
    let summary = read_tet_summary_v1(mmap)?;
    match summary.datasets.iter().position(|d| d.name == doc.dataset) {
        Some(dataset_idx) => plan_query_matched_dataset(
            &summary,
            dataset_idx,
            doc,
            mmap,
            tet_path,
            raw_f32_preview_max,
            spill_allowlist,
        ),
        None => Ok(plan_query_unmatched_dataset(doc, tet_path, &summary)),
    }
}

fn plan_query_matched_dataset(
    summary: &TetFileSummaryV1,
    dataset_idx: usize,
    doc: &QueryDocument,
    mmap: &[u8],
    tet_path: Option<&str>,
    raw_f32_preview_max: Option<usize>,
    spill_allowlist: Option<&SpillPathAllowlist>,
) -> Result<QueryResponse, TetError> {
    let rows = summary
        .chunks
        .iter()
        .filter(|c| c.dataset_id == dataset_idx as u64)
        .count();
    let rec = &summary.datasets[dataset_idx];
    let ndim = rec.shape.len();
    let (g0, g1, steps) = resolved_dense_global_box(doc, &rec.shape)?;
    let strided = steps.iter().any(|&t| t != 1);
    let touch_policy = if strided {
        CHUNK_TOUCH_POLICY.strided_half_open
    } else {
        CHUNK_TOUCH_POLICY.dense_half_open_unit_step
    };
    let coords = chunk_coords_intersecting_strided(&rec.shape, &rec.chunk_shape, &g0, &g1, &steps)
        .map_err(map_geometry_catalog_error)?;
    let read_plan = build_read_plan(
        summary,
        dataset_idx,
        ndim,
        &coords,
        touch_policy,
        &ReadPlanGeometry::new(&rec.shape, &rec.chunk_shape, &g0, &g1, &steps),
    )?;
    if doc.operation.is_some() && raw_f32_preview_max.is_none() {
        return Err(TetError::Validation(
            "`operation` requires mmap execution with an explicit preview limit (e.g. `--execute --preview-f32 64`, or `--preview-f32 0` to omit preview floats)".into(),
        ));
    }
    let mut message = String::from(
        "query accepted; dataset matched; read_plan lists mmap payload regions for touched chunks",
    );
    let execution = match raw_f32_preview_max {
        None => None,
        Some(0) if doc.operation.is_none() => {
            return Err(TetError::Validation(
                "preview limit 0 without `operation` would attach an empty execution block; omit `--execute` or use a positive `--preview-f32`".into(),
            ));
        }
        Some(limit) => {
            let default_spill;
            let spill_ref = match spill_allowlist {
                Some(p) => Some(p),
                None => {
                    if let Some(tp) = tet_path {
                        default_spill = SpillPathAllowlist::default_for_tet(
                            Path::new(tp),
                            std::iter::empty::<PathBuf>(),
                        )?;
                        Some(&default_spill)
                    } else {
                        None
                    }
                }
            };
            let preview = build_execution_preview(&super::operations::ExecutionPreviewInput {
                mmap,
                plan: &read_plan,
                dtype: rec.dtype,
                operation: doc.operation.as_ref(),
                output: doc.output.as_ref(),
                max_f32: limit,
                budget: ExecutionBudget::resolve(&summary.file_execution, doc.execution.as_ref()),
                spill_allowlist: spill_ref,
            })?;
            if doc.operation.is_some() {
                message.push_str(
                    "; operation executed (see execution.memory_strategy and operation_*)",
                );
            } else {
                message.push_str("; mmap f32 preview attached (see execution)");
            }
            Some(preview)
        }
    };
    Ok(query_response(
        doc,
        message,
        tet_path,
        Some(DatasetResolution {
            matched: true,
            dataset_index: Some(dataset_idx),
            dtype: Some(rec.dtype),
            shape: Some(rec.shape.clone()),
            chunk_shape: Some(rec.chunk_shape.clone()),
            chunk_index_rows: Some(rows),
            dataset_f32_bytes: (rec.dtype == DTYPE_F32)
                .then(|| f32_tensor_bytes_from_shape(&rec.shape))
                .flatten(),
            file_execution: Some(summary.file_execution),
            available_datasets: None,
        }),
        Some(read_plan),
        execution,
    ))
}

fn plan_query_unmatched_dataset(
    doc: &QueryDocument,
    tet_path: Option<&str>,
    summary: &TetFileSummaryV1,
) -> QueryResponse {
    let names: Vec<String> = summary.datasets.iter().map(|d| d.name.clone()).collect();
    query_response(
        doc,
        "query accepted; dataset name not found in this file (see catalog.available_datasets)",
        tet_path,
        Some(DatasetResolution {
            matched: false,
            dataset_index: None,
            dtype: None,
            shape: None,
            chunk_shape: None,
            chunk_index_rows: None,
            dataset_f32_bytes: None,
            file_execution: Some(summary.file_execution),
            available_datasets: Some(names),
        }),
        None,
        None,
    )
}
