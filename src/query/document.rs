//! Parse and validate JSON query documents.
//!
//! Security boundaries and deployment guidance: see `docs/query_engine.md` (section
//! “JSON security (input and output)”).

use crate::catalog::MAX_NDIM;

use super::document_wire::parse_query_value;
use super::types::{AxisSlice, Operation, QueryDocument, TetError};

/// Input limits enforced by [`parse_query_json`] and [`validate_query`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryLimits {
    /// Maximum query JSON payload size accepted by [`parse_query_json`].
    pub max_json_bytes: usize,
    /// Maximum composite nesting depth (arrays/objects) in query JSON.
    pub max_json_depth: usize,
    /// Maximum length of the `dataset` name string (bytes).
    pub max_dataset_name_len: usize,
    /// Maximum per-axis operation label length (decimal indices are short).
    pub max_operation_axis_label_len: usize,
}

impl QueryLimits {
    /// Default limits used by the query engine.
    pub const DEFAULT: Self = Self {
        max_json_bytes: 1 << 20,
        max_json_depth: 64,
        max_dataset_name_len: 1024,
        max_operation_axis_label_len: 32,
    };
}

/// Composite nesting depth of a parsed JSON value (objects/arrays only).
fn json_composite_depth(value: &serde_json::Value) -> usize {
    match value {
        serde_json::Value::Array(items) => {
            1 + items.iter().map(json_composite_depth).max().unwrap_or(0)
        }
        serde_json::Value::Object(map) => {
            1 + map.values().map(json_composite_depth).max().unwrap_or(0)
        }
        _ => 0,
    }
}

/// Parse a JSON query document from `text`.
///
/// # Errors
///
/// Returns [`TetError::Validation`] when `text` exceeds [`QueryLimits::DEFAULT`].
/// Returns [`TetError::InvalidJson`] when `text` is not valid JSON or does not deserialize into a
/// [`QueryDocument`] (including unknown object keys).
pub fn parse_query_json(text: &str) -> Result<QueryDocument, TetError> {
    let limits = QueryLimits::DEFAULT;
    if text.len() > limits.max_json_bytes {
        return Err(TetError::Validation(format!(
            "query JSON exceeds maximum size ({} bytes, limit {})",
            text.len(),
            limits.max_json_bytes
        )));
    }
    let value: serde_json::Value = serde_json::from_str(text)?;
    let depth = json_composite_depth(&value);
    if depth > limits.max_json_depth {
        return Err(TetError::Validation(format!(
            "query JSON nesting depth {depth} exceeds maximum {}",
            limits.max_json_depth
        )));
    }
    parse_query_value(&value)
}

/// JSON-only checks: `step != 0`, and when both `start` and `stop` are set, `start < stop`.
pub(super) fn validate_axis_slice_json(i: usize, sl: &AxisSlice) -> Result<(), TetError> {
    if let Some(step) = sl.step
        && step == 0
    {
        return Err(TetError::Validation(format!(
            "selection[{i}].step must be >= 1, got 0"
        )));
    }
    match (sl.start, sl.stop) {
        (Some(a), Some(b)) if a >= b => Err(TetError::Validation(format!(
            "selection[{i}]: start must be < stop when both set (got {a} >= {b})"
        ))),
        _ => Ok(()),
    }
}

/// Validate a parsed query document.
///
/// # Errors
///
/// Returns [`TetError::Validation`] when required fields or slice semantics are invalid.
pub fn validate_query(doc: &QueryDocument) -> Result<(), TetError> {
    let limits = QueryLimits::DEFAULT;
    let name = doc.dataset.trim();
    if name.is_empty() {
        return Err(TetError::Validation(
            "`dataset` must be a non-empty string".into(),
        ));
    }
    if doc.dataset.len() > limits.max_dataset_name_len {
        return Err(TetError::Validation(format!(
            "`dataset` name exceeds maximum length ({} > {})",
            doc.dataset.len(),
            limits.max_dataset_name_len
        )));
    }
    if let Some(axes) = &doc.selection {
        if axes.len() > MAX_NDIM {
            return Err(TetError::Validation(format!(
                "selection rank {} exceeds maximum {MAX_NDIM}",
                axes.len()
            )));
        }
        for (i, sl) in axes.iter().enumerate() {
            validate_axis_slice_json(i, sl)?;
        }
    }
    if let Some(op) = &doc.operation {
        validate_operation_axes_v1(op)?;
        validate_operation_params(op)?;
    }
    Ok(())
}

fn validate_operation_params(op: &Operation) -> Result<(), TetError> {
    match op {
        Operation::Quantile { q, .. } if !(0.0..=1.0).contains(q) => Err(TetError::Validation(
            format!("quantile q must be in [0.0, 1.0], got {q}"),
        )),
        Operation::Histogram { bins, .. } if *bins == 0 => {
            Err(TetError::Validation("histogram bins must be >= 1".into()))
        }
        Operation::Histogram { bins, .. } if *bins > 4096 => Err(TetError::Validation(format!(
            "histogram bins exceeds maximum 4096 (got {bins})"
        ))),
        _ => Ok(()),
    }
}

fn validate_operation_axis_token(s: &str) -> Result<(), TetError> {
    let max_label = QueryLimits::DEFAULT.max_operation_axis_label_len;
    if s.is_empty() {
        return Err(TetError::Validation(
            "operation axis label must not be empty".into(),
        ));
    }
    if s.len() > max_label {
        return Err(TetError::Validation(format!(
            "operation axis label exceeds maximum length ({} > {max_label})",
            s.len()
        )));
    }
    if !s.chars().all(|c| c.is_ascii_digit()) {
        return Err(TetError::Validation(
            "operation axes must be decimal dimension indices (e.g. \"0\"); non-ASCII labels are not supported yet".into(),
        ));
    }
    Ok(())
}

fn validate_operation_axes_v1(op: &Operation) -> Result<(), TetError> {
    let axes = op.axes();
    if axes.len() > MAX_NDIM {
        return Err(TetError::Validation(format!(
            "operation has {} axes; maximum is {MAX_NDIM}",
            axes.len()
        )));
    }
    for a in axes {
        validate_operation_axis_token(a)?;
    }
    Ok(())
}
