//! Query planning against a mmap’d `.tet` and optional raw `f32` preview reads.

type ResolvedGlobalBox = (Vec<u64>, Vec<u64>, Vec<u64>);

use crate::catalog::{
    CatalogError, ChunkIndexEntryV1, DTYPE_F32, MAX_NDIM, TetFileSummaryV1,
    chunk_coords_intersecting_strided, read_tet_summary_v1,
};

use super::document::validate_axis_slice_json;
use super::types::{
    CHUNK_TOUCH_POLICY, DatasetResolution, PlannedChunkIo, QueryDocument, QueryExecutionPreview,
    QueryResponse, ReadPlan, TetError,
};

fn resolved_dense_global_box(
    doc: &QueryDocument,
    shape: &[u64],
) -> Result<ResolvedGlobalBox, TetError> {
    let ndim = shape.len();
    if ndim == 0 {
        return Err(TetError::Validation(
            "dataset rank must be at least 1 for selection planning".into(),
        ));
    }
    match &doc.selection {
        None => Ok((vec![0u64; ndim], shape.to_vec(), vec![1u64; ndim])),
        Some(axes) => {
            if axes.len() != ndim {
                return Err(TetError::Validation(format!(
                    "selection must specify exactly {ndim} axes (one per dataset dimension), got {}",
                    axes.len()
                )));
            }
            let mut g0 = Vec::with_capacity(ndim);
            let mut g1 = Vec::with_capacity(ndim);
            let mut steps = Vec::with_capacity(ndim);
            for (d, sl) in axes.iter().enumerate() {
                validate_axis_slice_json(d, sl)?;
                let sd = shape[d];
                let start = sl.start.unwrap_or(0);
                let stop = sl.stop.unwrap_or(sd);
                if start >= sd {
                    return Err(TetError::Validation(format!(
                        "selection[{d}].start must be < shape[{d}] ({sd}), got {start}"
                    )));
                }
                if stop > sd {
                    return Err(TetError::Validation(format!(
                        "selection[{d}].stop must be <= shape[{d}] ({sd}), got {stop}"
                    )));
                }
                if start >= stop {
                    return Err(TetError::Validation(format!(
                        "selection[{d}]: require start < stop (got {start} >= {stop})"
                    )));
                }
                g0.push(start);
                g1.push(stop);
                steps.push(sl.step.unwrap_or(1));
            }
            Ok((g0, g1, steps))
        }
    }
}

fn planned_chunk_io(
    ndim: usize,
    coord: &[u64; MAX_NDIM],
    entry: &ChunkIndexEntryV1,
) -> PlannedChunkIo {
    PlannedChunkIo {
        chunk_index: coord[..ndim].to_vec(),
        payload_offset: entry.payload_offset,
        stored_byte_len: entry.stored_byte_len,
        raw_byte_len: entry.raw_byte_len,
        codec: entry.codec,
    }
}

fn find_chunk_entry<'a>(
    summary: &'a TetFileSummaryV1,
    dataset_idx: usize,
    ndim: usize,
    coord: &[u64; MAX_NDIM],
) -> Option<&'a ChunkIndexEntryV1> {
    summary.chunks.iter().find(|c| {
        c.dataset_id == dataset_idx as u64 && (0..ndim).all(|d| c.chunk_index[d] == coord[d])
    })
}

fn build_read_plan(
    summary: &TetFileSummaryV1,
    dataset_idx: usize,
    ndim: usize,
    coords: &[[u64; MAX_NDIM]],
    chunk_touch_policy: &'static str,
) -> Result<ReadPlan, TetError> {
    let mut chunks = Vec::with_capacity(coords.len());
    let mut total_stored: u64 = 0;
    for coord in coords {
        let entry = find_chunk_entry(summary, dataset_idx, ndim, coord).ok_or_else(|| {
            TetError::Validation(format!(
                "chunk index has no row for dataset_id={dataset_idx} chunk_index={:?}",
                &coord[..ndim]
            ))
        })?;
        total_stored = total_stored
            .checked_add(entry.stored_byte_len)
            .ok_or_else(|| {
                TetError::Validation("total stored bytes overflow when summing read plan".into())
            })?;
        chunks.push(planned_chunk_io(ndim, coord, entry));
    }
    Ok(ReadPlan {
        chunk_touch_policy,
        chunk_count: chunks.len(),
        total_stored_bytes: total_stored,
        chunks,
    })
}

fn u64_to_usize(field: &'static str, v: u64) -> Result<usize, TetError> {
    usize::try_from(v)
        .map_err(|_| TetError::Validation(format!("{field}={v} is too large for this platform")))
}

/// Map each planned chunk to a subslice of `mmap` (zero-copy).
///
/// # Errors
///
/// Returns [`TetError::Validation`] when a chunk uses a non-raw codec, lengths disagree, ranges
/// overflow, or payload bytes fall outside `mmap`.
pub fn planned_chunk_mmap_slices<'a>(
    mmap: &'a [u8],
    plan: &ReadPlan,
) -> Result<Vec<&'a [u8]>, TetError> {
    let mut out = Vec::with_capacity(plan.chunks.len());
    for c in &plan.chunks {
        if c.codec != 0 {
            return Err(TetError::Validation(format!(
                "only raw codec 0 is readable; got codec={} for chunk_index={:?}",
                c.codec, c.chunk_index
            )));
        }
        if c.stored_byte_len != c.raw_byte_len {
            return Err(TetError::Validation(format!(
                "raw codec requires stored_byte_len == raw_byte_len for chunk_index={:?}",
                c.chunk_index
            )));
        }
        let off = u64_to_usize("payload_offset", c.payload_offset)?;
        let len = u64_to_usize("stored_byte_len", c.stored_byte_len)?;
        let end = off
            .checked_add(len)
            .ok_or_else(|| TetError::Validation("payload byte range overflow".into()))?;
        if end > mmap.len() {
            return Err(TetError::Validation(format!(
                "payload [{off},{end}) extends past mmap length {}",
                mmap.len()
            )));
        }
        out.push(&mmap[off..end]);
    }
    Ok(out)
}

/// Decode planned raw `f32` chunk payloads (little-endian) in [`ReadPlan::chunks`] order.
///
/// `max_elements`: `None` decodes every float in the planned payloads (caller must ensure this
/// fits memory). `Some(n)` returns at most `n` values and sets `truncated` when more floats would
/// follow.
///
/// # Errors
///
/// Returns [`TetError::Validation`] when chunks are not raw `f32`-compatible (codec, byte
/// lengths), mmap bounds fail, or `max_elements` is `Some(0)`.
pub fn materialize_read_plan_f32_le(
    mmap: &[u8],
    plan: &ReadPlan,
    max_elements: Option<usize>,
) -> Result<(Vec<f32>, bool, u64), TetError> {
    if matches!(max_elements, Some(0)) {
        return Err(TetError::Validation(
            "max_elements must be at least 1 when Some".into(),
        ));
    }
    let slices = planned_chunk_mmap_slices(mmap, plan)?;
    let total_bytes_read_from_disk: u64 = slices.iter().try_fold(0u64, |acc, s| {
        acc.checked_add(s.len() as u64)
            .ok_or_else(|| TetError::Validation("total bytes read overflow".into()))
    })?;
    let mut out = Vec::new();
    let mut truncated = false;
    'outer: for sl in &slices {
        if sl.len() % 4 != 0 {
            return Err(TetError::Validation(format!(
                "chunk payload length {} is not a multiple of 4 for f32",
                sl.len()
            )));
        }
        for quad in sl.chunks_exact(4) {
            if let Some(cap) = max_elements
                && out.len() >= cap
            {
                truncated = true;
                break 'outer;
            }
            let a: [u8; 4] = quad
                .try_into()
                .map_err(|_| TetError::Validation("internal: f32 slice chunking".into()))?;
            out.push(f32::from_le_bytes(a));
        }
    }
    Ok((out, truncated, total_bytes_read_from_disk))
}

fn build_execution_preview(
    mmap: &[u8],
    plan: &ReadPlan,
    dtype: u32,
    max_f32: usize,
) -> Result<QueryExecutionPreview, TetError> {
    if dtype != DTYPE_F32 {
        return Err(TetError::Validation(
            "f32 preview requires dataset dtype f32 (DTYPE_F32 = 1)".into(),
        ));
    }
    let (f32_preview, f32_preview_truncated, total_bytes_read_from_disk) =
        materialize_read_plan_f32_le(mmap, plan, Some(max_f32))?;
    Ok(QueryExecutionPreview {
        total_bytes_read_from_disk,
        f32_preview,
        f32_preview_truncated,
    })
}

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

/// Build a response echoing the plan (no `.tet` on disk consulted).
#[must_use]
pub fn plan_query(doc: &QueryDocument) -> QueryResponse {
    base_response(
        doc,
        "query accepted and validated; pass `--tet PATH` for catalog + read_plan, or add `--execute` with `--tet` for a capped raw f32 mmap preview",
    )
}

/// Like [`plan_query`], but read catalog metadata from `mmap` (full `.tet` bytes).
///
/// `tet_path` is echoed in the JSON as `tet_file` when `Some`.
///
/// When the dataset matches, [`QueryResponse::read_plan`] lists chunk payloads that intersect the
/// per-axis selection (default full tensor). Non-unit JSON `step` uses strided chunk selection
/// (see [`crate::query::CHUNK_TOUCH_POLICY`]).
///
/// `raw_f32_preview_max`: when `Some(n)` with `n > 0`, mmap-read planned raw chunks and attach up
/// to `n` decoded little-endian `f32` values under [`QueryResponse::execution`] (requires dataset
/// [`DTYPE_F32`](crate::catalog::DTYPE_F32)).
///
/// # Errors
///
/// Returns [`TetError::Catalog`] when the file is not a valid v1 catalog view, or
/// [`TetError::Validation`] when `selection` is inconsistent with dataset rank or shape, when
/// preview is requested for a non-`f32` dataset, or when payload bytes cannot be read from `mmap`.
pub fn plan_query_with_tet_mmap(
    doc: &QueryDocument,
    tet_path: Option<&str>,
    mmap: &[u8],
    raw_f32_preview_max: Option<usize>,
) -> Result<QueryResponse, TetError> {
    let summary = read_tet_summary_v1(mmap)?;

    let idx = summary.datasets.iter().position(|d| d.name == doc.dataset);
    if let Some(i) = idx {
        let rows = summary
            .chunks
            .iter()
            .filter(|c| c.dataset_id == i as u64)
            .count();
        let rec = &summary.datasets[i];
        let ndim = rec.shape.len();
        let (g0, g1, steps) = resolved_dense_global_box(doc, &rec.shape)?;
        let strided = steps.iter().any(|&t| t != 1);
        let touch_policy = if strided {
            CHUNK_TOUCH_POLICY.strided_half_open
        } else {
            CHUNK_TOUCH_POLICY.dense_half_open_unit_step
        };
        let coords =
            chunk_coords_intersecting_strided(&rec.shape, &rec.chunk_shape, &g0, &g1, &steps)
                .map_err(map_geometry_catalog_error)?;
        let read_plan = build_read_plan(&summary, i, ndim, &coords, touch_policy)?;
        let mut message = String::from(
            "query accepted; dataset matched; read_plan lists mmap payload regions (aggregations not executed)",
        );
        if doc.operation.is_some() {
            message.push_str("; operation field ignored in this build");
        }
        let execution = if let Some(limit) = raw_f32_preview_max.filter(|&n| n > 0) {
            let preview = build_execution_preview(mmap, &read_plan, rec.dtype, limit)?;
            message.push_str("; mmap f32 preview attached (see execution)");
            Some(preview)
        } else {
            None
        };
        return Ok(query_response(
            doc,
            message,
            tet_path,
            Some(DatasetResolution {
                matched: true,
                dataset_index: Some(i),
                dtype: Some(rec.dtype),
                shape: Some(rec.shape.clone()),
                chunk_shape: Some(rec.chunk_shape.clone()),
                chunk_index_rows: Some(rows),
                available_datasets: None,
            }),
            Some(read_plan),
            execution,
        ));
    }

    let names: Vec<String> = summary.datasets.iter().map(|d| d.name.clone()).collect();
    Ok(query_response(
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
            available_datasets: Some(names),
        }),
        None,
        None,
    ))
}
