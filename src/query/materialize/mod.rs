//! Decode planned chunk payloads into logical row-major tensors (f32/f64/i32/i64).

pub mod int;
pub mod parallel;
pub mod stats;

use std::path::Path;

use memmap2::Mmap;

use crate::query::decode::chunk_decode::{
    fold_planned_chunk_f32, scatter_chunk_into_plan, scatter_chunk_into_plan_f64,
    visit_planned_chunk_f64,
};
use crate::query::engine::budget::{ExecutionBudget, MemoryStrategy};
use crate::query::engine::spill_policy::{SpillPathAllowlist, TempSpillFile};
pub(crate) use crate::query::fold::FoldPlanOutcome;
use crate::query::fold::reduction::{ArgIndexAccum, ReductionKind, ValueAccum};
use crate::query::fold::shared::{
    FoldPreviewBuffer, build_fold_plan_outcome, build_fold_plan_outcome_typed,
    validate_fold_preview,
};
use crate::query::types::{ReadPlan, TetError};
use crate::utils::dtype::ElementDtype;
use crate::utils::f64_le;
use parallel::{materialize_read_plan_f32_le_parallel, materialize_read_plan_f64_le_parallel};

type ScatterFillFn = fn(&[u8], &ReadPlan, &mut [f32]) -> Result<u64, TetError>;

pub(crate) fn materialize_read_plan_f32_le_core(
    mmap: &[u8],
    plan: &ReadPlan,
    max_elements: Option<usize>,
    scatter_fill: ScatterFillFn,
) -> Result<(Vec<f32>, bool, u64), TetError> {
    if matches!(max_elements, Some(0)) {
        return Ok((Vec::new(), false, 0));
    }
    let n = plan.logical_f32_element_count;
    let cap = max_elements.map(|m| m.min(n));
    let (buf_len, truncated) = match cap {
        Some(c) if c < n => (c, true),
        _ => (n, max_elements.is_some_and(|m| m < n)),
    };
    let mut out = vec![f32::NAN; buf_len];
    let total_bytes_read_from_disk = scatter_fill(mmap, plan, &mut out)?;
    check_materialized_complete(&out)?;
    Ok((out, truncated, total_bytes_read_from_disk))
}

pub(crate) fn materialize_read_plan_f32_le_into_core(
    mmap: &[u8],
    plan: &ReadPlan,
    max_elements: Option<usize>,
    dst: &mut [f32],
    scatter_fill: ScatterFillFn,
) -> Result<MaterializeReadPlanF32IntoOutcome, TetError> {
    let n = plan.logical_f32_element_count;
    if matches!(max_elements, Some(0)) {
        return Ok(MaterializeReadPlanF32IntoOutcome {
            logical_element_count: n,
            elements_written: 0,
            truncated: n > 0,
            total_bytes_read_from_disk: 0,
        });
    }
    let want_write = max_elements.map_or(n, |m| m.min(n));
    if dst.len() < want_write {
        return Err(TetError::Validation(format!(
            "destination buffer length {} < required {} (logical element count {})",
            dst.len(),
            want_write,
            n
        )));
    }
    let total_bytes_read_from_disk = if want_write < n {
        let mut tmp = vec![f32::NAN; want_write];
        let bytes = scatter_fill(mmap, plan, &mut tmp)?;
        check_materialized_complete(&tmp)?;
        dst[..want_write].copy_from_slice(&tmp);
        bytes
    } else {
        let bytes = scatter_fill(mmap, plan, &mut dst[..n])?;
        check_materialized_complete(&dst[..n])?;
        bytes
    };
    Ok(MaterializeReadPlanF32IntoOutcome {
        logical_element_count: n,
        elements_written: want_write,
        truncated: max_elements.is_some_and(|m| m < n),
        total_bytes_read_from_disk,
    })
}

fn check_materialized_complete(out: &[f32]) -> Result<(), TetError> {
    if out.iter().any(|v| v.is_nan()) {
        return Err(TetError::Validation(
            "materialized selection has unset elements (chunk payloads vs selection mismatch)"
                .into(),
        ));
    }
    Ok(())
}

/// Decode planned raw `f32` chunk payloads (little-endian) into **logical row-major** order for the
/// strided selection encoded on [`ReadPlan`].
///
/// `max_elements`: `None` decodes every float in the logical tensor. `Some(0)` returns an empty
/// vector and reads nothing from disk. `Some(n)` for `n > 0` returns the first `n` values in
/// logical row-major order and sets `truncated` when the logical tensor is longer.
///
/// # Errors
///
/// Returns [`TetError::Validation`] when chunk payloads disagree with tile geometry, the
/// strided selection is not fully covered by planned chunks, mmap bounds fail, or zstd decode
/// fails.
pub fn materialize_read_plan_f32_le(
    mmap: &[u8],
    plan: &ReadPlan,
    max_elements: Option<usize>,
) -> Result<(Vec<f32>, bool, u64), TetError> {
    materialize_read_plan_f32_le_core(mmap, plan, max_elements, materialize_scatter_fill)
}

/// Outcome of [`materialize_read_plan_f32_le_into`].
#[derive(Debug, Clone)]
pub struct MaterializeReadPlanF32IntoOutcome {
    /// Logical tensor element count (selection grid product).
    pub logical_element_count: usize,
    /// Values written to the start of the caller buffer (`min(max_elements.unwrap_or(logical), logical)`).
    pub elements_written: usize,
    pub truncated: bool,
    pub total_bytes_read_from_disk: u64,
}

/// Like [`materialize_read_plan_f32_le`], but writes decoded values into `dst` without allocating a `Vec`.
///
/// When `max_elements` is `None`, `dst.len()` must be at least [`ReadPlan::logical_f32_element_count`].
/// When `max_elements` is `Some(m)` with `m > 0`, `dst.len()` must be at least `m.min(logical)`.
/// `Some(0)` writes nothing and does not touch `dst`.
///
/// # Errors
///
/// Same failure modes as [`materialize_read_plan_f32_le`], plus a short destination buffer.
pub fn materialize_read_plan_f32_le_into(
    mmap: &[u8],
    plan: &ReadPlan,
    max_elements: Option<usize>,
    dst: &mut [f32],
) -> Result<MaterializeReadPlanF32IntoOutcome, TetError> {
    materialize_read_plan_f32_le_into_core(mmap, plan, max_elements, dst, materialize_scatter_fill)
}

pub(crate) fn validate_read_plan_geometry(plan: &ReadPlan, out_len: usize) -> Result<(), TetError> {
    let ndim = plan.dataset_shape.len();
    if plan.chunk_shape.len() != ndim
        || plan.selection_box_start.len() != ndim
        || plan.selection_box_stop_exclusive.len() != ndim
        || plan.selection_step.len() != ndim
        || plan.logical_selection_shape.len() != ndim
    {
        return Err(TetError::Validation(
            "read_plan geometry fields have inconsistent rank".into(),
        ));
    }
    if out_len > plan.logical_f32_element_count {
        return Err(TetError::Validation(format!(
            "output buffer length {out_len} exceeds read_plan.logical_f32_element_count {}",
            plan.logical_f32_element_count
        )));
    }
    Ok(())
}

pub(crate) fn validate_full_read_plan_buffer(
    plan: &ReadPlan,
    out_len: usize,
) -> Result<(), TetError> {
    validate_read_plan_geometry(plan, out_len)?;
    if out_len != plan.logical_f32_element_count {
        return Err(TetError::Validation(format!(
            "output buffer length {out_len} != read_plan.logical_f32_element_count {}",
            plan.logical_f32_element_count
        )));
    }
    Ok(())
}

/// Decode planned chunks once, aggregating a scalar reduction without allocating the full
/// logical tensor. Fills `f32_preview` with the first `max_f32` logical row-major values.
///
/// # Errors
///
/// Same validation failures as materialization when chunk payloads disagree with the plan.
pub(crate) fn fold_read_plan_scalar_operation(
    mmap: &[u8],
    plan: &ReadPlan,
    max_f32: usize,
    kind: ReductionKind,
    policy: &crate::query::fold::fold_policy::FoldIoPolicy,
) -> Result<FoldPlanOutcome, TetError> {
    if crate::query::fold::parallel_fold::use_parallel_fold(plan, policy) {
        return crate::query::fold::parallel_fold::fold_read_plan_scalar_operation_parallel(
            mmap,
            plan,
            max_f32,
            kind,
            policy.fold_workers,
        );
    }
    let n = plan.logical_f32_element_count;
    let preview_cap = max_f32.min(n);
    let mut preview = vec![f32::NAN; preview_cap];
    let mut total_bytes_read_from_disk: u64 = 0;
    let chunk_order =
        crate::query::fold::fold_policy::chunk_indices_for_fold(plan, policy.sequential_io);

    let operation = match kind {
        ReductionKind::ArgMin | ReductionKind::ArgMax => {
            let mut acc = ArgIndexAccum::default();
            for i in chunk_order {
                let c = &plan.chunks[i];
                let chunk_bytes = fold_planned_chunk_f32(
                    mmap,
                    plan,
                    c,
                    kind,
                    &mut ValueAccum::default(),
                    &mut acc,
                    &mut preview,
                )?;
                total_bytes_read_from_disk = total_bytes_read_from_disk
                    .checked_add(chunk_bytes)
                    .ok_or_else(|| TetError::Validation("total bytes read overflow".into()))?;
            }
            validate_fold_preview(!acc.is_empty(), &preview, preview_cap)?;
            acc.finish_scalar(kind, n).into()
        }
        _ => {
            let mut acc = ValueAccum::default();
            let mut arg = ArgIndexAccum::default();
            for i in chunk_order {
                let c = &plan.chunks[i];
                let chunk_bytes =
                    fold_planned_chunk_f32(mmap, plan, c, kind, &mut acc, &mut arg, &mut preview)?;
                total_bytes_read_from_disk = total_bytes_read_from_disk
                    .checked_add(chunk_bytes)
                    .ok_or_else(|| TetError::Validation("total bytes read overflow".into()))?;
            }
            validate_fold_preview(!acc.is_empty(), &preview, preview_cap)?;
            acc.finish_scalar(kind).into()
        }
    };

    Ok(build_fold_plan_outcome(
        preview,
        max_f32,
        n,
        total_bytes_read_from_disk,
        operation,
    ))
}

pub(crate) fn materialize_scatter_fill(
    mmap: &[u8],
    plan: &ReadPlan,
    out: &mut [f32],
) -> Result<u64, TetError> {
    validate_read_plan_geometry(plan, out.len())?;
    let mut total_bytes_read_from_disk: u64 = 0;
    for c in &plan.chunks {
        let n = scatter_chunk_into_plan(mmap, plan, c, out)?;
        total_bytes_read_from_disk = total_bytes_read_from_disk
            .checked_add(n)
            .ok_or_else(|| TetError::Validation("total bytes read overflow".into()))?;
    }
    Ok(total_bytes_read_from_disk)
}

/// Spill the full logical selection as row-major `f32` LE to `path` using a file-backed mmap
/// (disk-resident; does not allocate a dense `Vec` in RAM).
///
/// # Errors
///
/// Same validation failures as [`materialize_read_plan_f32_le`], plus I/O or mmap errors on `path`.
pub fn spill_read_plan_f32_le(
    mmap: &[u8],
    plan: &ReadPlan,
    path: &std::path::Path,
) -> Result<u64, TetError> {
    use std::fs::OpenOptions;
    use std::io::Write;

    use memmap2::MmapMut;

    let n = plan.logical_f32_element_count;
    let byte_len = u64::try_from(n)
        .map_err(|_| TetError::Validation("logical element count overflow".into()))?;
    let byte_len = crate::utils::f32_le::bytes_from_elem_count(byte_len)
        .ok_or_else(|| TetError::Validation("spill byte length overflow".into()))?;
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .map_err(|e| TetError::Validation(format!("spill open failed: {e}")))?;
    file.set_len(byte_len)
        .map_err(|e| TetError::Validation(format!("spill set_len failed: {e}")))?;
    file.flush()
        .map_err(|e| TetError::Validation(format!("spill flush failed: {e}")))?;
    let mut out_mmap = unsafe {
        MmapMut::map_mut(&file)
            .map_err(|e| TetError::Validation(format!("spill mmap failed: {e}")))?
    };
    let out = bytemuck::cast_slice_mut(out_mmap.as_mut());
    validate_full_read_plan_buffer(plan, out.len())?;
    let total = materialize_scatter_fill(mmap, plan, out)?;
    check_materialized_complete(out)?;
    out_mmap
        .flush()
        .map_err(|e| TetError::Validation(format!("spill mmap flush failed: {e}")))?;
    Ok(total)
}

// --- Full logical selection (RAM or engine temp spill) for tier-C ops ---

pub(crate) enum LogicalF32Backing {
    InMemory(Vec<f32>),
    TempSpill(TempSpillFile),
}

pub(crate) enum LogicalF64Backing {
    InMemory(Vec<f64>),
    TempSpill(TempSpillFile),
}

/// Capped decode previews for all supported element types.
#[derive(Debug, Clone, Default)]
#[allow(clippy::struct_excessive_bools)]
pub(crate) struct DecodePreviewBundle {
    pub f32: Vec<f32>,
    pub f64: Vec<f64>,
    pub i32: Vec<i32>,
    pub i64: Vec<i64>,
    pub f32_truncated: bool,
    pub f64_truncated: bool,
    pub i32_truncated: bool,
    pub i64_truncated: bool,
}

impl DecodePreviewBundle {
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn all_truncated(truncated: bool) -> Self {
        Self {
            f32_truncated: truncated,
            f64_truncated: truncated,
            i32_truncated: truncated,
            i64_truncated: truncated,
            ..Self::default()
        }
    }
}

pub(crate) enum MaterializedLogical {
    F32 {
        backing: LogicalF32Backing,
        total_bytes_read_from_disk: u64,
        strategy: MemoryStrategy,
    },
    F64 {
        backing: LogicalF64Backing,
        total_bytes_read_from_disk: u64,
        strategy: MemoryStrategy,
    },
    I32 {
        backing: int::LogicalI32Backing,
        total_bytes_read_from_disk: u64,
        strategy: MemoryStrategy,
    },
    I64 {
        backing: int::LogicalI64Backing,
        total_bytes_read_from_disk: u64,
        strategy: MemoryStrategy,
    },
}

fn materialize_into_vec(mmap: &[u8], plan: &ReadPlan) -> Result<(Vec<f32>, u64), TetError> {
    if plan.chunks.len() <= 1 {
        materialize_read_plan_f32_le(mmap, plan, None)
    } else {
        materialize_read_plan_f32_le_parallel(mmap, plan, None)
    }
    .map(|(v, truncated, bytes)| {
        debug_assert!(!truncated);
        (v, bytes)
    })
}

fn materialize_into_vec_f64(mmap: &[u8], plan: &ReadPlan) -> Result<(Vec<f64>, u64), TetError> {
    if plan.chunks.len() <= 1 {
        materialize_read_plan_f64_le(mmap, plan, None)
    } else {
        materialize_read_plan_f64_le_parallel(mmap, plan, None)
    }
    .map(|(v, truncated, bytes)| {
        debug_assert!(!truncated);
        (v, bytes)
    })
}

/// Decode the full logical selection, choosing RAM vs temp spill from the memory budget.
pub(crate) fn materialize_logical_selection(
    mmap: &[u8],
    plan: &ReadPlan,
    budget: &ExecutionBudget,
    allowlist: &SpillPathAllowlist,
    dtype: ElementDtype,
) -> Result<MaterializedLogical, TetError> {
    if budget.full_tensor_exceeds_budget(plan, dtype)? {
        let temp = TempSpillFile::create(allowlist)?;
        let bytes = crate::query::dispatch::spill_full_selection(mmap, plan, temp.path(), dtype)?;
        Ok(match dtype {
            ElementDtype::F32 => MaterializedLogical::F32 {
                backing: LogicalF32Backing::TempSpill(temp),
                total_bytes_read_from_disk: bytes,
                strategy: MemoryStrategy::TempSpillMaterialize,
            },
            ElementDtype::F64 => MaterializedLogical::F64 {
                backing: LogicalF64Backing::TempSpill(temp),
                total_bytes_read_from_disk: bytes,
                strategy: MemoryStrategy::TempSpillMaterialize,
            },
            ElementDtype::I32 => MaterializedLogical::I32 {
                backing: int::LogicalI32Backing::TempSpill(temp),
                total_bytes_read_from_disk: bytes,
                strategy: MemoryStrategy::TempSpillMaterialize,
            },
            ElementDtype::I64 => MaterializedLogical::I64 {
                backing: int::LogicalI64Backing::TempSpill(temp),
                total_bytes_read_from_disk: bytes,
                strategy: MemoryStrategy::TempSpillMaterialize,
            },
        })
    } else {
        match dtype {
            ElementDtype::F32 => {
                let (vec, bytes) = materialize_into_vec(mmap, plan)?;
                Ok(MaterializedLogical::F32 {
                    backing: LogicalF32Backing::InMemory(vec),
                    total_bytes_read_from_disk: bytes,
                    strategy: MemoryStrategy::InMemoryMaterialize,
                })
            }
            ElementDtype::F64 => {
                let (vec, bytes) = materialize_into_vec_f64(mmap, plan)?;
                Ok(MaterializedLogical::F64 {
                    backing: LogicalF64Backing::InMemory(vec),
                    total_bytes_read_from_disk: bytes,
                    strategy: MemoryStrategy::InMemoryMaterialize,
                })
            }
            ElementDtype::I32 => {
                let (vec, bytes) = int::materialize_into_vec_i32(mmap, plan)?;
                Ok(MaterializedLogical::I32 {
                    backing: int::LogicalI32Backing::InMemory(vec),
                    total_bytes_read_from_disk: bytes,
                    strategy: MemoryStrategy::InMemoryMaterialize,
                })
            }
            ElementDtype::I64 => {
                let (vec, bytes) = int::materialize_into_vec_i64(mmap, plan)?;
                Ok(MaterializedLogical::I64 {
                    backing: int::LogicalI64Backing::InMemory(vec),
                    total_bytes_read_from_disk: bytes,
                    strategy: MemoryStrategy::InMemoryMaterialize,
                })
            }
        }
    }
}

/// First `max` logical values without a second full decode (from RAM or spill file).
pub(crate) fn preview_from_materialized(
    materialized: &MaterializedLogical,
    logical_len: usize,
    max: usize,
) -> Result<DecodePreviewBundle, TetError> {
    let cap = max.min(logical_len);
    if cap == 0 {
        return Ok(DecodePreviewBundle::all_truncated(logical_len > 0));
    }
    match materialized {
        MaterializedLogical::F32 { backing, .. } => {
            let (p, t) = preview_from_materialized_f32(backing, logical_len, max)?;
            Ok(DecodePreviewBundle {
                f32: p,
                f32_truncated: t,
                ..DecodePreviewBundle::empty()
            })
        }
        MaterializedLogical::F64 { backing, .. } => {
            let (p, t) = preview_from_materialized_f64(backing, logical_len, max)?;
            Ok(DecodePreviewBundle {
                f64: p,
                f64_truncated: t,
                ..DecodePreviewBundle::empty()
            })
        }
        MaterializedLogical::I32 { backing, .. } => {
            let (p, t) = int::preview_from_materialized_i32(backing, logical_len, max)?;
            Ok(DecodePreviewBundle {
                i32: p,
                i32_truncated: t,
                ..DecodePreviewBundle::empty()
            })
        }
        MaterializedLogical::I64 { backing, .. } => {
            let (p, t) = int::preview_from_materialized_i64(backing, logical_len, max)?;
            Ok(DecodePreviewBundle {
                i64: p,
                i64_truncated: t,
                ..DecodePreviewBundle::empty()
            })
        }
    }
}

fn preview_from_materialized_f32(
    backing: &LogicalF32Backing,
    logical_len: usize,
    max_f32: usize,
) -> Result<(Vec<f32>, bool), TetError> {
    let cap = max_f32.min(logical_len);
    if cap == 0 {
        return Ok((Vec::new(), logical_len > 0));
    }
    match backing {
        LogicalF32Backing::InMemory(v) => Ok((v[..cap].to_vec(), logical_len > max_f32)),
        LogicalF32Backing::TempSpill(temp) => {
            preview_from_spill_file_f32(temp.path(), cap, logical_len)
        }
    }
}

fn preview_from_materialized_f64(
    backing: &LogicalF64Backing,
    logical_len: usize,
    max: usize,
) -> Result<(Vec<f64>, bool), TetError> {
    let cap = max.min(logical_len);
    if cap == 0 {
        return Ok((Vec::new(), logical_len > 0));
    }
    match backing {
        LogicalF64Backing::InMemory(v) => Ok((v[..cap].to_vec(), logical_len > max)),
        LogicalF64Backing::TempSpill(temp) => {
            preview_from_spill_file_f64(temp.path(), cap, logical_len)
        }
    }
}

fn preview_from_spill_file_f32(
    path: &Path,
    cap: usize,
    logical_len: usize,
) -> Result<(Vec<f32>, bool), TetError> {
    let file = std::fs::File::open(path)
        .map_err(|e| TetError::Validation(format!("temp spill read failed: {e}")))?;
    let mmap = unsafe {
        Mmap::map(&file)
            .map_err(|e| TetError::Validation(format!("temp spill mmap failed: {e}")))?
    };
    let slice: &[f32] = bytemuck::cast_slice(&mmap);
    if slice.len() < cap {
        return Err(TetError::Validation(
            "temp spill shorter than logical selection".into(),
        ));
    }
    Ok((slice[..cap].to_vec(), logical_len > cap))
}

fn preview_from_spill_file_f64(
    path: &Path,
    cap: usize,
    logical_len: usize,
) -> Result<(Vec<f64>, bool), TetError> {
    let file = std::fs::File::open(path)
        .map_err(|e| TetError::Validation(format!("temp spill read failed: {e}")))?;
    let mmap = unsafe {
        Mmap::map(&file)
            .map_err(|e| TetError::Validation(format!("temp spill mmap failed: {e}")))?
    };
    let slice: &[f64] = bytemuck::cast_slice(&mmap);
    if slice.len() < cap {
        return Err(TetError::Validation(
            "temp spill shorter than logical selection".into(),
        ));
    }
    Ok((slice[..cap].to_vec(), logical_len > cap))
}

/// Preview from an export spill file (single decode pass for spill + preview).
pub(crate) fn preview_from_spill_export_file(
    path: &Path,
    logical_len: usize,
    max: usize,
    dtype: ElementDtype,
) -> Result<DecodePreviewBundle, TetError> {
    let cap = max.min(logical_len);
    match dtype {
        ElementDtype::F32 => {
            let (p, t) = preview_from_spill_file_f32(path, cap, logical_len)?;
            Ok(DecodePreviewBundle {
                f32: p,
                f32_truncated: t,
                ..DecodePreviewBundle::empty()
            })
        }
        ElementDtype::F64 => {
            let (p, t) = preview_from_spill_file_f64(path, cap, logical_len)?;
            Ok(DecodePreviewBundle {
                f64: p,
                f64_truncated: t,
                ..DecodePreviewBundle::empty()
            })
        }
        ElementDtype::I32 => {
            let (p, t) = int::preview_from_spill_file_i32(path, cap, logical_len)?;
            Ok(DecodePreviewBundle {
                i32: p,
                i32_truncated: t,
                ..DecodePreviewBundle::empty()
            })
        }
        ElementDtype::I64 => {
            let (p, t) = int::preview_from_spill_file_i64(path, cap, logical_len)?;
            Ok(DecodePreviewBundle {
                i64: p,
                i64_truncated: t,
                ..DecodePreviewBundle::empty()
            })
        }
    }
}

// --- f64 materialize / fold / spill ---

type ScatterFillF64Fn = fn(&[u8], &ReadPlan, &mut [f64]) -> Result<u64, TetError>;

fn check_materialized_complete_f64(out: &[f64]) -> Result<(), TetError> {
    if out.iter().any(|v| v.is_nan()) {
        return Err(TetError::Validation(
            "materialized selection has unset elements (chunk payloads vs selection mismatch)"
                .into(),
        ));
    }
    Ok(())
}

pub(crate) fn materialize_read_plan_f64_le_core(
    mmap: &[u8],
    plan: &ReadPlan,
    max_elements: Option<usize>,
    scatter_fill: ScatterFillF64Fn,
) -> Result<(Vec<f64>, bool, u64), TetError> {
    if matches!(max_elements, Some(0)) {
        return Ok((Vec::new(), false, 0));
    }
    let n = plan.logical_f32_element_count;
    let cap = max_elements.map(|m| m.min(n));
    let (buf_len, truncated) = match cap {
        Some(c) if c < n => (c, true),
        _ => (n, max_elements.is_some_and(|m| m < n)),
    };
    let mut out = vec![f64::NAN; buf_len];
    let total_bytes_read_from_disk = scatter_fill(mmap, plan, &mut out)?;
    check_materialized_complete_f64(&out)?;
    Ok((out, truncated, total_bytes_read_from_disk))
}

/// Decode planned raw `f64` chunk payloads (little-endian) into **logical row-major** order for the
/// strided selection encoded on [`ReadPlan`].
///
/// `max_elements`: `None` decodes every float in the logical tensor. `Some(0)` returns an empty
/// vector and reads nothing from disk. `Some(n)` for `n > 0` returns the first `n` values in
/// logical row-major order and sets `truncated` when the logical tensor is longer.
///
/// # Errors
///
/// Returns [`TetError::Validation`] when chunk payloads disagree with tile geometry, the
/// strided selection is not fully covered by planned chunks, mmap bounds fail, or zstd decode
/// fails.
pub fn materialize_read_plan_f64_le(
    mmap: &[u8],
    plan: &ReadPlan,
    max_elements: Option<usize>,
) -> Result<(Vec<f64>, bool, u64), TetError> {
    materialize_read_plan_f64_le_core(mmap, plan, max_elements, materialize_scatter_fill_f64)
}

pub(crate) fn materialize_scatter_fill_f64(
    mmap: &[u8],
    plan: &ReadPlan,
    out: &mut [f64],
) -> Result<u64, TetError> {
    validate_read_plan_geometry(plan, out.len())?;
    let mut total_bytes_read_from_disk: u64 = 0;
    for c in &plan.chunks {
        let n = scatter_chunk_into_plan_f64(mmap, plan, c, out)?;
        total_bytes_read_from_disk = total_bytes_read_from_disk
            .checked_add(n)
            .ok_or_else(|| TetError::Validation("total bytes read overflow".into()))?;
    }
    Ok(total_bytes_read_from_disk)
}

/// Spill the full logical selection as row-major `f64` LE to `path` using a file-backed mmap
/// (disk-resident; does not allocate a dense `Vec` in RAM).
///
/// # Errors
///
/// Same validation failures as [`materialize_read_plan_f64_le`], plus I/O or mmap errors on `path`.
pub fn spill_read_plan_f64_le(
    mmap: &[u8],
    plan: &ReadPlan,
    path: &std::path::Path,
) -> Result<u64, TetError> {
    use std::fs::OpenOptions;
    use std::io::Write;

    use memmap2::MmapMut;

    let n = plan.logical_f32_element_count;
    let byte_len = u64::try_from(n)
        .map_err(|_| TetError::Validation("logical element count overflow".into()))?;
    let byte_len = f64_le::bytes_from_elem_count(byte_len)
        .ok_or_else(|| TetError::Validation("spill byte length overflow".into()))?;
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .map_err(|e| TetError::Validation(format!("spill open failed: {e}")))?;
    file.set_len(byte_len)
        .map_err(|e| TetError::Validation(format!("spill set_len failed: {e}")))?;
    file.flush()
        .map_err(|e| TetError::Validation(format!("spill flush failed: {e}")))?;
    let mut out_mmap = unsafe {
        MmapMut::map_mut(&file)
            .map_err(|e| TetError::Validation(format!("spill mmap failed: {e}")))?
    };
    let out = bytemuck::cast_slice_mut(out_mmap.as_mut());
    validate_full_read_plan_buffer(plan, out.len())?;
    let total = materialize_scatter_fill_f64(mmap, plan, out)?;
    check_materialized_complete_f64(out)?;
    out_mmap
        .flush()
        .map_err(|e| TetError::Validation(format!("spill mmap flush failed: {e}")))?;
    Ok(total)
}

pub(crate) fn fold_read_plan_scalar_operation_f64(
    mmap: &[u8],
    plan: &ReadPlan,
    max_preview: usize,
    kind: ReductionKind,
    policy: &crate::query::fold::fold_policy::FoldIoPolicy,
) -> Result<FoldPlanOutcome, TetError> {
    if crate::query::fold::parallel_fold::use_parallel_fold(plan, policy) {
        return crate::query::fold::parallel_fold::fold_read_plan_scalar_operation_f64_parallel(
            mmap,
            plan,
            max_preview,
            kind,
            policy.fold_workers,
        );
    }
    let n = plan.logical_f32_element_count;
    let preview_cap = max_preview.min(n);
    let mut f64_preview = vec![f64::NAN; preview_cap];
    let mut total_bytes_read_from_disk: u64 = 0;
    let chunk_order =
        crate::query::fold::fold_policy::chunk_indices_for_fold(plan, policy.sequential_io);

    let operation = match kind {
        ReductionKind::ArgMin | ReductionKind::ArgMax => {
            let mut acc = ArgIndexAccum::default();
            for i in chunk_order {
                let c = &plan.chunks[i];
                let chunk_bytes = visit_planned_chunk_f64(mmap, plan, c, |li, v| {
                    acc.push_f64(li as u64, v, kind);
                    if li < preview_cap {
                        f64_preview[li] = v;
                    }
                    Ok(())
                })?;
                total_bytes_read_from_disk = total_bytes_read_from_disk
                    .checked_add(chunk_bytes)
                    .ok_or_else(|| TetError::Validation("total bytes read overflow".into()))?;
            }
            if acc.is_empty() {
                return Err(TetError::Validation(
                    "operation requires at least one decoded value from the read plan".into(),
                ));
            }
            if preview_cap > 0 && f64_preview.iter().any(|v| v.is_nan()) {
                return Err(TetError::Validation(
                    "materialized selection has unset preview elements".into(),
                ));
            }
            acc.finish_scalar(kind, n).into()
        }
        _ => {
            let mut acc = ValueAccum::default();
            for i in chunk_order {
                let c = &plan.chunks[i];
                let chunk_bytes = visit_planned_chunk_f64(mmap, plan, c, |li, v| {
                    acc.push_f64(v);
                    if li < preview_cap {
                        f64_preview[li] = v;
                    }
                    Ok(())
                })?;
                total_bytes_read_from_disk = total_bytes_read_from_disk
                    .checked_add(chunk_bytes)
                    .ok_or_else(|| TetError::Validation("total bytes read overflow".into()))?;
            }
            if acc.is_empty() {
                return Err(TetError::Validation(
                    "operation requires at least one decoded value from the read plan".into(),
                ));
            }
            if preview_cap > 0 && f64_preview.iter().any(|v| v.is_nan()) {
                return Err(TetError::Validation(
                    "materialized selection has unset preview elements".into(),
                ));
            }
            acc.finish_scalar(kind).into()
        }
    };

    Ok(build_fold_plan_outcome_typed(
        FoldPreviewBuffer::F64(f64_preview),
        max_preview,
        n,
        total_bytes_read_from_disk,
        operation,
    ))
}
