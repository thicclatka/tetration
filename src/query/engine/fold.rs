//! Shared helpers for chunk-streaming fold operations.

use crate::query::types::{OperationPreviewFields, TetError};

/// Result of scalar or streaming partial fold operations.
#[derive(Debug, Clone)]
pub(crate) struct FoldPlanOutcome {
    pub f32_preview: Vec<f32>,
    pub f64_preview: Vec<f64>,
    pub i32_preview: Vec<i32>,
    pub i64_preview: Vec<i64>,
    pub f32_preview_truncated: bool,
    pub f64_preview_truncated: bool,
    pub i32_preview_truncated: bool,
    pub i64_preview_truncated: bool,
    pub total_bytes_read_from_disk: u64,
    pub operation: OperationPreviewFields,
}

/// Validate that a fold saw data and filled a preview buffer when required.
pub(crate) fn validate_fold_preview(
    saw_any: bool,
    preview: &[f32],
    preview_cap: usize,
) -> Result<(), TetError> {
    if !saw_any {
        return Err(TetError::Validation(
            "operation requires at least one decoded f32 from the read plan".into(),
        ));
    }
    if preview_cap > 0 && preview.iter().any(|v| v.is_nan()) {
        return Err(TetError::Validation(
            "materialized selection has unset preview elements (chunk payloads vs selection mismatch)"
                .into(),
        ));
    }
    Ok(())
}

/// Build a [`FoldPlanOutcome`] after preview validation.
#[must_use]
pub(crate) fn build_fold_plan_outcome(
    preview: Vec<f32>,
    max_f32: usize,
    logical_len: usize,
    total_bytes_read_from_disk: u64,
    operation: OperationPreviewFields,
) -> FoldPlanOutcome {
    FoldPlanOutcome {
        f32_preview: if max_f32 == 0 { Vec::new() } else { preview },
        f64_preview: Vec::new(),
        i32_preview: Vec::new(),
        i64_preview: Vec::new(),
        f32_preview_truncated: logical_len > max_f32,
        f64_preview_truncated: false,
        i32_preview_truncated: false,
        i64_preview_truncated: false,
        total_bytes_read_from_disk,
        operation,
    }
}
