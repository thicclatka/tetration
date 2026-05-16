//! Parse and validate JSON query documents.

use super::types::{AxisSlice, Operation, QueryDocument, TetError};

/// Parse a JSON query document from `text`.
///
/// # Errors
///
/// Returns [`TetError::InvalidJson`] when `text` is not valid JSON or does not deserialize into a
/// [`QueryDocument`].
pub fn parse_query_json(text: &str) -> Result<QueryDocument, TetError> {
    Ok(serde_json::from_str(text)?)
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
    if doc.dataset.trim().is_empty() {
        return Err(TetError::Validation(
            "`dataset` must be a non-empty string".into(),
        ));
    }
    if let Some(axes) = &doc.selection {
        for (i, sl) in axes.iter().enumerate() {
            validate_axis_slice_json(i, sl)?;
        }
    }
    if let Some(op) = &doc.operation {
        validate_operation_axes_v1(op)?;
    }
    Ok(())
}

fn validate_operation_axis_token(s: &str) -> Result<(), TetError> {
    if s.is_empty() {
        return Err(TetError::Validation(
            "operation axis label must not be empty".into(),
        ));
    }
    if !s.chars().all(|c| c.is_ascii_digit()) {
        return Err(TetError::Validation(
            "operation axes must be decimal dimension indices (e.g. \"0\"); non-ASCII labels are not supported yet".into(),
        ));
    }
    Ok(())
}

fn validate_operation_axes_v1(op: &Operation) -> Result<(), TetError> {
    for a in op.axes() {
        validate_operation_axis_token(a)?;
    }
    Ok(())
}
