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
    pub u32_preview: Vec<u32>,
    pub u64_preview: Vec<u64>,
    pub f16_preview: Vec<half::f16>,
    pub f32_preview_truncated: bool,
    pub f64_preview_truncated: bool,
    pub i32_preview_truncated: bool,
    pub i64_preview_truncated: bool,
    pub u8_preview_truncated: bool,
    pub u16_preview_truncated: bool,
    pub i16_preview_truncated: bool,
    pub u32_preview_truncated: bool,
    pub u64_preview_truncated: bool,
    pub f16_preview_truncated: bool,
    pub total_bytes_read_from_disk: u64,
    pub operation: OperationPreviewFields,
}

/// Active preview buffer for a fold result (one dtype per outcome).
pub(crate) enum FoldPreviewBuffer {
    F32(Vec<f32>),
    F64(Vec<f64>),
    F16(Vec<half::f16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
}

#[derive(Default)]
struct EmptyPreviews {
    f32: Vec<f32>,
    f64: Vec<f64>,
    f16: Vec<half::f16>,
    i32: Vec<i32>,
    i64: Vec<i64>,
    u8: Vec<u8>,
    u16: Vec<u16>,
    i16: Vec<i16>,
    u32: Vec<u32>,
    u64: Vec<u64>,
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
    let mut e = EmptyPreviews::default();
    let mut f32_t = false;
    let mut f64_t = false;
    let mut f16_t = false;
    let mut i32_t = false;
    let mut i64_t = false;
    let mut u8_t = false;
    let mut u16_t = false;
    let mut i16_t = false;
    let mut u32_t = false;
    let mut u64_t = false;

    match preview {
        FoldPreviewBuffer::F32(v) => {
            e.f32 = if empty { Vec::new() } else { v };
            f32_t = truncated;
        }
        FoldPreviewBuffer::F64(v) => {
            e.f64 = if empty { Vec::new() } else { v };
            f64_t = truncated;
        }
        FoldPreviewBuffer::F16(v) => {
            e.f16 = if empty { Vec::new() } else { v };
            f16_t = truncated;
        }
        FoldPreviewBuffer::I32(v) => {
            e.i32 = if empty { Vec::new() } else { v };
            i32_t = truncated;
        }
        FoldPreviewBuffer::I64(v) => {
            e.i64 = if empty { Vec::new() } else { v };
            i64_t = truncated;
        }
        FoldPreviewBuffer::U8(v) => {
            e.u8 = if empty { Vec::new() } else { v };
            u8_t = truncated;
        }
        FoldPreviewBuffer::U16(v) => {
            e.u16 = if empty { Vec::new() } else { v };
            u16_t = truncated;
        }
        FoldPreviewBuffer::I16(v) => {
            e.i16 = if empty { Vec::new() } else { v };
            i16_t = truncated;
        }
        FoldPreviewBuffer::U32(v) => {
            e.u32 = if empty { Vec::new() } else { v };
            u32_t = truncated;
        }
        FoldPreviewBuffer::U64(v) => {
            e.u64 = if empty { Vec::new() } else { v };
            u64_t = truncated;
        }
    }

    FoldPlanOutcome {
        f32_preview: e.f32,
        f64_preview: e.f64,
        f16_preview: e.f16,
        i32_preview: e.i32,
        i64_preview: e.i64,
        u8_preview: e.u8,
        u16_preview: e.u16,
        i16_preview: e.i16,
        u32_preview: e.u32,
        u64_preview: e.u64,
        f32_preview_truncated: f32_t,
        f64_preview_truncated: f64_t,
        f16_preview_truncated: f16_t,
        i32_preview_truncated: i32_t,
        i64_preview_truncated: i64_t,
        u8_preview_truncated: u8_t,
        u16_preview_truncated: u16_t,
        i16_preview_truncated: i16_t,
        u32_preview_truncated: u32_t,
        u64_preview_truncated: u64_t,
        total_bytes_read_from_disk,
        operation,
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

/// Like [`validate_fold_preview_f64`] for `f16` preview buffers.
pub(crate) fn validate_fold_preview_f16(
    saw_any: bool,
    preview: &[half::f16],
    preview_cap: usize,
) -> Result<(), TetError> {
    validate_fold_preview_unset(
        saw_any,
        preview_cap,
        preview.iter().any(|v| v.is_nan()),
        "operation requires at least one decoded value from the read plan",
    )
}
