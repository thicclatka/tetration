//! Sequential byte-stream scalar fold over one contiguous raw payload span (hyperslab path).

use crate::catalog::CHUNK_PAYLOAD_CODEC_V1;
use crate::query::fold::reduction::{ReductionKind, ValueAccum};
use crate::query::fold::shared::{
    FoldPreviewBuffer, build_fold_plan_outcome_typed, validate_fold_preview,
    validate_fold_preview_f64,
};
use crate::query::types::{ReadPlan, TetError};
use crate::utils::dtype::ElementDtype;
use crate::utils::wire;

/// Raw payload bytes per sequential scan window (64 MiB — matches bench slab size).
pub const SCAN_WINDOW_RAW_BYTES: usize = 64 * 1024 * 1024;

/// Contiguous raw byte range covering the full logical selection in on-disk order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContiguousRawSpan {
    pub start: usize,
    pub len: usize,
}

/// Tier-A/B scalar ops supported without per-element logical indices.
#[must_use]
pub fn supports_scalar_kind(kind: ReductionKind) -> bool {
    !matches!(kind, ReductionKind::ArgMin | ReductionKind::ArgMax)
}

/// Full dense scan with raw codec and adjacent payloads in logical plan order.
#[must_use]
pub fn detect_contiguous_raw_span(plan: &ReadPlan, elem_size: usize) -> Option<ContiguousRawSpan> {
    if plan.chunks.is_empty() {
        return None;
    }
    let expected_raw = plan.logical_f32_element_count.checked_mul(elem_size)? as u64;
    if expected_raw == 0 {
        return None;
    }
    for c in &plan.chunks {
        if c.codec != CHUNK_PAYLOAD_CODEC_V1.raw || c.stored_byte_len != c.raw_byte_len {
            return None;
        }
    }
    for w in plan.chunks.windows(2) {
        let end = w[0]
            .payload_offset
            .checked_add(w[0].raw_byte_len)
            .filter(|&end| end == w[1].payload_offset)?;
        let _ = end;
    }
    let total_raw: u64 = plan.chunks.iter().map(|c| c.raw_byte_len).sum();
    if total_raw != expected_raw {
        return None;
    }
    let start = usize::try_from(plan.chunks[0].payload_offset).ok()?;
    let len = usize::try_from(total_raw).ok()?;
    Some(ContiguousRawSpan { start, len })
}

fn span_slice(mmap: &[u8], span: ContiguousRawSpan) -> Result<&[u8], TetError> {
    let range = wire::checked_usize_subslice(span.start, span.len, mmap.len()).map_err(|e| {
        TetError::Validation(format!("contiguous payload span out of mmap bounds: {e:?}"))
    })?;
    Ok(&mmap[range])
}

fn fill_f32_preview(raw: &[u8], preview: &mut [f32], global_offset_elems: usize) -> bool {
    let vals: &[f32] = bytemuck::cast_slice(raw);
    let mut wrote = false;
    for (k, &v) in vals.iter().enumerate() {
        let li = global_offset_elems + k;
        if li < preview.len() {
            preview[li] = v;
            wrote = true;
        } else {
            break;
        }
    }
    wrote
}

fn fill_f64_preview(raw: &[u8], preview: &mut [f64], global_offset_elems: usize) -> bool {
    let vals: &[f64] = bytemuck::cast_slice(raw);
    let mut wrote = false;
    for (k, &v) in vals.iter().enumerate() {
        let li = global_offset_elems + k;
        if li < preview.len() {
            preview[li] = v;
            wrote = true;
        } else {
            break;
        }
    }
    wrote
}

fn fill_i32_preview(raw: &[u8], preview: &mut [i32], global_offset_elems: usize) -> bool {
    let vals: &[i32] = bytemuck::cast_slice(raw);
    let mut wrote = false;
    for (k, &v) in vals.iter().enumerate() {
        let li = global_offset_elems + k;
        if li < preview.len() {
            preview[li] = v;
            wrote = true;
        } else {
            break;
        }
    }
    wrote
}

fn fill_i64_preview(raw: &[u8], preview: &mut [i64], global_offset_elems: usize) -> bool {
    let vals: &[i64] = bytemuck::cast_slice(raw);
    let mut wrote = false;
    for (k, &v) in vals.iter().enumerate() {
        let li = global_offset_elems + k;
        if li < preview.len() {
            preview[li] = v;
            wrote = true;
        } else {
            break;
        }
    }
    wrote
}

fn fold_window_f32(
    acc: &mut ValueAccum,
    window: &[u8],
    kind: ReductionKind,
    preview: &mut [f32],
    global_offset_elems: usize,
) -> bool {
    debug_assert_eq!(window.len() % 4, 0);
    acc.push_f32_le_bytes(window, kind);
    if preview.is_empty() {
        false
    } else {
        fill_f32_preview(window, preview, global_offset_elems)
    }
}

fn fold_window_f64(
    acc: &mut ValueAccum,
    window: &[u8],
    kind: ReductionKind,
    preview: &mut [f64],
    global_offset_elems: usize,
) -> bool {
    debug_assert_eq!(window.len() % 8, 0);
    acc.push_f64_le_bytes(window, kind);
    if preview.is_empty() {
        false
    } else {
        fill_f64_preview(window, preview, global_offset_elems)
    }
}

fn fold_window_i32(
    acc: &mut ValueAccum,
    window: &[u8],
    kind: ReductionKind,
    preview: &mut [i32],
    global_offset_elems: usize,
) -> bool {
    debug_assert_eq!(window.len() % 4, 0);
    if matches!(kind, ReductionKind::Count) {
        acc.push_f32_le_bytes(window, kind);
    } else {
        let vals: &[i32] = bytemuck::cast_slice(window);
        for &v in vals {
            acc.push_f64(f64::from(v));
        }
    }
    if preview.is_empty() {
        false
    } else {
        fill_i32_preview(window, preview, global_offset_elems)
    }
}

fn fold_window_i64(
    acc: &mut ValueAccum,
    window: &[u8],
    kind: ReductionKind,
    preview: &mut [i64],
    global_offset_elems: usize,
) -> bool {
    debug_assert_eq!(window.len() % 8, 0);
    if matches!(kind, ReductionKind::Count) {
        acc.push_f64_le_bytes(window, kind);
    } else {
        let vals: &[i64] = bytemuck::cast_slice(window);
        for &v in vals {
            acc.push_f64(v as f64);
        }
    }
    if preview.is_empty() {
        false
    } else {
        fill_i64_preview(window, preview, global_offset_elems)
    }
}

fn fold_span_windows(
    raw: &[u8],
    elem_size: usize,
    mut fold_window: impl FnMut(&mut ValueAccum, &[u8], usize) -> bool,
) -> Result<(ValueAccum, bool), TetError> {
    let mut acc = ValueAccum::default();
    let mut offset = 0usize;
    let mut global_elems = 0usize;
    let mut saw_preview = false;
    while offset < raw.len() {
        let end = (offset + SCAN_WINDOW_RAW_BYTES).min(raw.len());
        let window = &raw[offset..end];
        if !window.len().is_multiple_of(elem_size) {
            return Err(TetError::Validation(
                "contiguous payload span length is not a multiple of element size".into(),
            ));
        }
        saw_preview |= fold_window(&mut acc, window, global_elems);
        offset = end;
        global_elems += window.len() / elem_size;
    }
    Ok((acc, saw_preview))
}

fn require_nonempty(acc: &ValueAccum) -> Result<(), TetError> {
    if acc.is_empty() {
        return Err(TetError::Validation(
            "operation requires at least one decoded value from the read plan".into(),
        ));
    }
    Ok(())
}

fn require_int_preview(preview_cap: usize, saw_preview: bool) -> Result<(), TetError> {
    if preview_cap > 0 && !saw_preview {
        return Err(TetError::Validation(
            "materialized selection has unset preview elements".into(),
        ));
    }
    Ok(())
}

pub(crate) fn fold_read_plan_scalar_linear(
    mmap: &[u8],
    plan: &ReadPlan,
    max_preview: usize,
    kind: ReductionKind,
    dtype: ElementDtype,
) -> Result<crate::query::fold::FoldPlanOutcome, TetError> {
    let elem_size = dtype.elem_size();
    let span = detect_contiguous_raw_span(plan, elem_size).ok_or_else(|| {
        TetError::Validation("linear scan requires contiguous raw payload span".into())
    })?;
    let n = plan.logical_f32_element_count;
    let preview_cap = max_preview.min(n);
    let raw = span_slice(mmap, span)?;
    let total_bytes = u64::try_from(span.len).unwrap_or(u64::MAX);

    match dtype {
        ElementDtype::F32 => {
            let mut preview = vec![f32::NAN; preview_cap];
            let (acc, _) = fold_span_windows(raw, elem_size, |acc, window, global_elems| {
                fold_window_f32(acc, window, kind, &mut preview, global_elems)
            })?;
            require_nonempty(&acc)?;
            validate_fold_preview(true, &preview, preview_cap)?;
            Ok(build_fold_plan_outcome_typed(
                FoldPreviewBuffer::F32(if max_preview == 0 {
                    Vec::new()
                } else {
                    preview
                }),
                max_preview,
                n,
                total_bytes,
                acc.finish_scalar(kind).into(),
            ))
        }
        ElementDtype::F64 => {
            let mut preview = vec![f64::NAN; preview_cap];
            let (acc, _) = fold_span_windows(raw, elem_size, |acc, window, global_elems| {
                fold_window_f64(acc, window, kind, &mut preview, global_elems)
            })?;
            require_nonempty(&acc)?;
            validate_fold_preview_f64(true, &preview, preview_cap)?;
            Ok(build_fold_plan_outcome_typed(
                FoldPreviewBuffer::F64(if max_preview == 0 {
                    Vec::new()
                } else {
                    preview
                }),
                max_preview,
                n,
                total_bytes,
                acc.finish_scalar(kind).into(),
            ))
        }
        ElementDtype::I32 => {
            let mut preview = vec![0i32; preview_cap];
            let (acc, saw_preview) =
                fold_span_windows(raw, elem_size, |acc, window, global_elems| {
                    fold_window_i32(acc, window, kind, &mut preview, global_elems)
                })?;
            require_nonempty(&acc)?;
            require_int_preview(preview_cap, saw_preview || preview_cap == 0)?;
            Ok(build_fold_plan_outcome_typed(
                FoldPreviewBuffer::I32(if max_preview == 0 {
                    Vec::new()
                } else {
                    preview
                }),
                max_preview,
                n,
                total_bytes,
                acc.finish_scalar(kind).into(),
            ))
        }
        ElementDtype::I64 => {
            let mut preview = vec![0i64; preview_cap];
            let (acc, saw_preview) =
                fold_span_windows(raw, elem_size, |acc, window, global_elems| {
                    fold_window_i64(acc, window, kind, &mut preview, global_elems)
                })?;
            require_nonempty(&acc)?;
            require_int_preview(preview_cap, saw_preview || preview_cap == 0)?;
            Ok(build_fold_plan_outcome_typed(
                FoldPreviewBuffer::I64(if max_preview == 0 {
                    Vec::new()
                } else {
                    preview
                }),
                max_preview,
                n,
                total_bytes,
                acc.finish_scalar(kind).into(),
            ))
        }
    }
}
