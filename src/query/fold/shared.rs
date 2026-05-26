//! Shared helpers for chunk-streaming fold operations.

use crate::query::types::{OperationPreviewFields, TetError};

/// Result of scalar or streaming partial fold operations.
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub(crate) struct FoldPlanOutcome {
    pub f32_preview: Vec<f32>,
    pub f64_preview: Vec<f64>,
    pub i32_preview: Vec<i32>,
    pub i64_preview: Vec<i64>,
    pub u8_preview: Vec<u8>,
    pub u16_preview: Vec<u16>,
    pub i16_preview: Vec<i16>,
    pub f32_preview_truncated: bool,
    pub f64_preview_truncated: bool,
    pub i32_preview_truncated: bool,
    pub i64_preview_truncated: bool,
    pub u8_preview_truncated: bool,
    pub u16_preview_truncated: bool,
    pub i16_preview_truncated: bool,
    pub total_bytes_read_from_disk: u64,
    pub operation: OperationPreviewFields,
}

/// Active preview buffer for a fold result (one dtype per outcome).
pub(crate) enum FoldPreviewBuffer {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
}

macro_rules! empty_int_previews {
    () => {
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new())
    };
}

/// Build a [`FoldPlanOutcome`] for any supported preview dtype.
#[must_use]
pub(crate) fn build_fold_plan_outcome_typed(
    preview: FoldPreviewBuffer,
    max_preview: usize,
    logical_len: usize,
    total_bytes_read_from_disk: u64,
    operation: OperationPreviewFields,
) -> FoldPlanOutcome {
    let empty = max_preview == 0;
    let truncated = logical_len > max_preview;
    let (i32_p, i64_p, u8_p, u16_p, i16_p) = empty_int_previews!();
    match preview {
        FoldPreviewBuffer::F32(v) => FoldPlanOutcome {
            f32_preview: if empty { Vec::new() } else { v },
            f64_preview: Vec::new(),
            i32_preview: i32_p,
            i64_preview: i64_p,
            u8_preview: u8_p,
            u16_preview: u16_p,
            i16_preview: i16_p,
            f32_preview_truncated: truncated,
            f64_preview_truncated: false,
            i32_preview_truncated: false,
            i64_preview_truncated: false,
            u8_preview_truncated: false,
            u16_preview_truncated: false,
            i16_preview_truncated: false,
            total_bytes_read_from_disk,
            operation,
        },
        FoldPreviewBuffer::F64(v) => FoldPlanOutcome {
            f32_preview: Vec::new(),
            f64_preview: if empty { Vec::new() } else { v },
            i32_preview: i32_p,
            i64_preview: i64_p,
            u8_preview: u8_p,
            u16_preview: u16_p,
            i16_preview: i16_p,
            f32_preview_truncated: false,
            f64_preview_truncated: truncated,
            i32_preview_truncated: false,
            i64_preview_truncated: false,
            u8_preview_truncated: false,
            u16_preview_truncated: false,
            i16_preview_truncated: false,
            total_bytes_read_from_disk,
            operation,
        },
        FoldPreviewBuffer::I32(v) => FoldPlanOutcome {
            f32_preview: Vec::new(),
            f64_preview: Vec::new(),
            i32_preview: if empty { Vec::new() } else { v },
            i64_preview: i64_p,
            u8_preview: u8_p,
            u16_preview: u16_p,
            i16_preview: i16_p,
            f32_preview_truncated: false,
            f64_preview_truncated: false,
            i32_preview_truncated: truncated,
            i64_preview_truncated: false,
            u8_preview_truncated: false,
            u16_preview_truncated: false,
            i16_preview_truncated: false,
            total_bytes_read_from_disk,
            operation,
        },
        FoldPreviewBuffer::I64(v) => FoldPlanOutcome {
            f32_preview: Vec::new(),
            f64_preview: Vec::new(),
            i32_preview: i32_p,
            i64_preview: if empty { Vec::new() } else { v },
            u8_preview: u8_p,
            u16_preview: u16_p,
            i16_preview: i16_p,
            f32_preview_truncated: false,
            f64_preview_truncated: false,
            i32_preview_truncated: false,
            i64_preview_truncated: truncated,
            u8_preview_truncated: false,
            u16_preview_truncated: false,
            i16_preview_truncated: false,
            total_bytes_read_from_disk,
            operation,
        },
        FoldPreviewBuffer::U8(v) => FoldPlanOutcome {
            f32_preview: Vec::new(),
            f64_preview: Vec::new(),
            i32_preview: i32_p,
            i64_preview: i64_p,
            u8_preview: if empty { Vec::new() } else { v },
            u16_preview: u16_p,
            i16_preview: i16_p,
            f32_preview_truncated: false,
            f64_preview_truncated: false,
            i32_preview_truncated: false,
            i64_preview_truncated: false,
            u8_preview_truncated: truncated,
            u16_preview_truncated: false,
            i16_preview_truncated: false,
            total_bytes_read_from_disk,
            operation,
        },
        FoldPreviewBuffer::U16(v) => FoldPlanOutcome {
            f32_preview: Vec::new(),
            f64_preview: Vec::new(),
            i32_preview: i32_p,
            i64_preview: i64_p,
            u8_preview: u8_p,
            u16_preview: if empty { Vec::new() } else { v },
            i16_preview: i16_p,
            f32_preview_truncated: false,
            f64_preview_truncated: false,
            i32_preview_truncated: false,
            i64_preview_truncated: false,
            u8_preview_truncated: false,
            u16_preview_truncated: truncated,
            i16_preview_truncated: false,
            total_bytes_read_from_disk,
            operation,
        },
        FoldPreviewBuffer::I16(v) => FoldPlanOutcome {
            f32_preview: Vec::new(),
            f64_preview: Vec::new(),
            i32_preview: i32_p,
            i64_preview: i64_p,
            u8_preview: u8_p,
            u16_preview: u16_p,
            i16_preview: if empty { Vec::new() } else { v },
            f32_preview_truncated: false,
            f64_preview_truncated: false,
            i32_preview_truncated: false,
            i64_preview_truncated: false,
            u8_preview_truncated: false,
            u16_preview_truncated: false,
            i16_preview_truncated: truncated,
            total_bytes_read_from_disk,
            operation,
        },
    }
}

/// Build a [`FoldPlanOutcome`] after preview validation (`f32` path).
#[must_use]
pub(crate) fn build_fold_plan_outcome(
    preview: Vec<f32>,
    max_f32: usize,
    logical_len: usize,
    total_bytes_read_from_disk: u64,
    operation: OperationPreviewFields,
) -> FoldPlanOutcome {
    build_fold_plan_outcome_typed(
        FoldPreviewBuffer::F32(preview),
        max_f32,
        logical_len,
        total_bytes_read_from_disk,
        operation,
    )
}

fn validate_fold_preview_unset(
    saw_any: bool,
    preview_cap: usize,
    has_unset: bool,
    empty_msg: &str,
) -> Result<(), TetError> {
    if !saw_any {
        return Err(TetError::Validation(empty_msg.into()));
    }
    if preview_cap > 0 && has_unset {
        return Err(TetError::Validation(
            "materialized selection has unset preview elements (chunk payloads vs selection mismatch)"
                .into(),
        ));
    }
    Ok(())
}

/// Validate that a fold saw data and filled a preview buffer when required.
pub(crate) fn validate_fold_preview(
    saw_any: bool,
    preview: &[f32],
    preview_cap: usize,
) -> Result<(), TetError> {
    validate_fold_preview_unset(
        saw_any,
        preview_cap,
        preview.iter().any(|v| v.is_nan()),
        "operation requires at least one decoded f32 from the read plan",
    )
}

/// Like [`validate_fold_preview`] for `f64` preview buffers.
pub(crate) fn validate_fold_preview_f64(
    saw_any: bool,
    preview: &[f64],
    preview_cap: usize,
) -> Result<(), TetError> {
    validate_fold_preview_unset(
        saw_any,
        preview_cap,
        preview.iter().any(|v| v.is_nan()),
        "operation requires at least one decoded value from the read plan",
    )
}
