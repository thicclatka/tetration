//! `i32` / `i64` logical materialize, spill, and fold helpers.

use std::path::Path;

use memmap2::{Mmap, MmapMut};

use crate::query::types::{PlannedChunkIo, ReadPlan, TetError};
use crate::utils::dtype::ElementDtype;
use crate::utils::{i32_le, i64_le};

use super::parallel::{
    materialize_scatter_fill_parallel_i32, materialize_scatter_fill_parallel_i64,
};
use crate::query::decode::chunk_decode::{
    scatter_chunk_into_plan_i32, scatter_chunk_into_plan_i64, visit_planned_chunk_i32_as_f64,
    visit_planned_chunk_i64_as_f64,
};
use crate::query::dispatch::accumulate_chunk_read_bytes;
use crate::query::engine::spill_policy::TempSpillFile;
use crate::query::fold::reduction::{ArgIndexAccum, ReductionKind, ValueAccum};
use crate::query::fold::shared::{
    FoldPlanOutcome, FoldPreviewBuffer, build_fold_plan_outcome_typed,
};

use super::{MaterializedLogical, validate_full_read_plan_buffer, validate_read_plan_geometry};

fn check_materialized_complete_option<T>(out: &[Option<T>]) -> Result<(), TetError> {
    if out.iter().any(Option::is_none) {
        return Err(TetError::Validation(
            "materialized selection has unset elements (chunk payloads vs selection mismatch)"
                .into(),
        ));
    }
    Ok(())
}

fn finalize_option<T: Copy + Default>(out: Vec<Option<T>>) -> Result<Vec<T>, TetError> {
    check_materialized_complete_option(&out)?;
    Ok(out.into_iter().map(Option::unwrap_or_default).collect())
}

fn materialize_read_plan_int_le_core<T, F>(
    mmap: &[u8],
    plan: &ReadPlan,
    max_elements: Option<usize>,
    scatter_fill: F,
) -> Result<(Vec<T>, bool, u64), TetError>
where
    T: Copy + Default,
    F: Fn(&[u8], &ReadPlan, &mut [Option<T>]) -> Result<u64, TetError>,
{
    if matches!(max_elements, Some(0)) {
        return Ok((Vec::new(), false, 0));
    }
    let n = plan.logical_f32_element_count;
    let cap = max_elements.map(|m| m.min(n));
    let (buf_len, truncated) = match cap {
        Some(c) if c < n => (c, true),
        _ => (n, max_elements.is_some_and(|m| m < n)),
    };
    let mut out = vec![None; buf_len];
    let total_bytes_read_from_disk = scatter_fill(mmap, plan, &mut out)?;
    let vec = finalize_option(out)?;
    Ok((vec, truncated, total_bytes_read_from_disk))
}

fn spill_read_plan_int_le_impl<T, F>(
    mmap: &[u8],
    plan: &ReadPlan,
    path: &Path,
    byte_len: u64,
    scatter: F,
) -> Result<u64, TetError>
where
    T: bytemuck::Pod + Copy + Default,
    F: Fn(&[u8], &ReadPlan, &mut [Option<T>]) -> Result<u64, TetError>,
{
    use std::fs::OpenOptions;
    use std::io::Write;

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
    let out: &mut [T] = bytemuck::cast_slice_mut(out_mmap.as_mut());
    validate_full_read_plan_buffer(plan, out.len())?;
    let mut slot = vec![None; out.len()];
    let total = scatter(mmap, plan, &mut slot)?;
    out.copy_from_slice(bytemuck::cast_slice(&finalize_option(slot)?));
    out_mmap
        .flush()
        .map_err(|e| TetError::Validation(format!("spill mmap flush failed: {e}")))?;
    Ok(total)
}

fn preview_from_backing_in_memory<T: Copy>(
    v: &[T],
    logical_len: usize,
    max: usize,
) -> (Vec<T>, bool) {
    let cap = max.min(logical_len);
    if cap == 0 {
        return (Vec::new(), logical_len > 0);
    }
    (v[..cap].to_vec(), logical_len > max)
}

fn preview_from_spill_file_pod<T: bytemuck::Pod + Copy>(
    path: &Path,
    cap: usize,
    logical_len: usize,
) -> Result<(Vec<T>, bool), TetError> {
    let file = std::fs::File::open(path)
        .map_err(|e| TetError::Validation(format!("temp spill read failed: {e}")))?;
    let mmap = unsafe {
        Mmap::map(&file)
            .map_err(|e| TetError::Validation(format!("temp spill mmap failed: {e}")))?
    };
    let slice: &[T] = bytemuck::cast_slice(&mmap);
    if slice.len() < cap {
        return Err(TetError::Validation(
            "temp spill shorter than logical selection".into(),
        ));
    }
    Ok((slice[..cap].to_vec(), logical_len > cap))
}

/// Per-dtype materialize/spill/preview surface (`i32` / `i64`).
macro_rules! define_int_materialize {
    (
        $elem:ty;
        backing $backing:ident;
        le_mod $le_mod:ident;
        scatter $scatter:path;
        scatter_seq $scatter_seq:ident;
        scatter_ty $scatter_ty:ident;
        scatter_par $scatter_par:path;
        core_fn $core_fn:ident;
        read_fn $read_fn:ident;
        spill_fn $spill_fn:ident;
        type_label $type_label:literal;
        into_vec_fn $into_vec_fn:ident;
        spill_file_fn $spill_file_fn:ident;
        preview_mat_fn $preview_mat_fn:ident;
        as_f64_fn $as_f64_fn:ident;
        promote_inmem |$v:ident| $promote_inmem:expr;
        promote_spill |$s:ident| $promote_spill:expr;
    ) => {
        pub(crate) enum $backing {
            InMemory(Vec<$elem>),
            TempSpill(TempSpillFile),
        }

        fn $scatter_seq(
            mmap: &[u8],
            plan: &ReadPlan,
            out: &mut [Option<$elem>],
        ) -> Result<u64, TetError> {
            validate_read_plan_geometry(plan, out.len())?;
            let mut total_bytes_read_from_disk: u64 = 0;
            for c in &plan.chunks {
                let n = $scatter(mmap, plan, c, out)?;
                accumulate_chunk_read_bytes(&mut total_bytes_read_from_disk, n)?;
            }
            Ok(total_bytes_read_from_disk)
        }

        type $scatter_ty = fn(&[u8], &ReadPlan, &mut [Option<$elem>]) -> Result<u64, TetError>;

        pub(crate) fn $core_fn(
            mmap: &[u8],
            plan: &ReadPlan,
            max_elements: Option<usize>,
            scatter_fill: $scatter_ty,
        ) -> Result<(Vec<$elem>, bool, u64), TetError> {
            materialize_read_plan_int_le_core(mmap, plan, max_elements, scatter_fill)
        }

        /// Decode planned raw [`$type_label`] chunk payloads (little-endian) into **logical row-major**
        /// order for the strided selection on [`ReadPlan`].
        ///
        /// `max_elements`: `None` decodes the full logical tensor. `Some(0)` returns an empty vector
        /// and reads nothing. `Some(n)` for `n > 0` returns the first `n` logical values and sets
        /// `truncated` when the logical tensor is longer.
        ///
        /// # Errors
        ///
        /// Returns [`TetError::Validation`] when chunk payloads disagree with tile geometry, the
        /// strided selection is not fully covered by planned chunks, or mmap bounds fail.
        pub fn $read_fn(
            mmap: &[u8],
            plan: &ReadPlan,
            max_elements: Option<usize>,
        ) -> Result<(Vec<$elem>, bool, u64), TetError> {
            $core_fn(mmap, plan, max_elements, $scatter_seq)
        }

        /// Spill the full logical selection as row-major [`$type_label`] LE to `path` via file-backed mmap.
        ///
        /// # Errors
        ///
        /// Same validation failures as [`$read_fn`], plus logical element count or spill byte length
        /// overflow, or I/O or mmap errors on `path`.
        pub fn $spill_fn(mmap: &[u8], plan: &ReadPlan, path: &Path) -> Result<u64, TetError> {
            let n = plan.logical_f32_element_count;
            let byte_len = u64::try_from(n)
                .map_err(|_| TetError::Validation("logical element count overflow".into()))?;
            let byte_len = $le_mod::bytes_from_elem_count(byte_len)
                .ok_or_else(|| TetError::Validation("spill byte length overflow".into()))?;
            spill_read_plan_int_le_impl(mmap, plan, path, byte_len, $scatter_seq)
        }

        pub(crate) fn $into_vec_fn(
            mmap: &[u8],
            plan: &ReadPlan,
        ) -> Result<(Vec<$elem>, u64), TetError> {
            let scatter = if plan.chunks.len() <= 1 {
                $scatter_seq
            } else {
                $scatter_par
            };
            $core_fn(mmap, plan, None, scatter).map(|(v, truncated, bytes)| {
                debug_assert!(!truncated);
                (v, bytes)
            })
        }

        pub(crate) fn $spill_file_fn(
            path: &Path,
            cap: usize,
            logical_len: usize,
        ) -> Result<(Vec<$elem>, bool), TetError> {
            preview_from_spill_file_pod(path, cap, logical_len)
        }

        pub(crate) fn $preview_mat_fn(
            backing: &$backing,
            logical_len: usize,
            max: usize,
        ) -> Result<(Vec<$elem>, bool), TetError> {
            match backing {
                $backing::InMemory(v) => Ok(preview_from_backing_in_memory(v, logical_len, max)),
                $backing::TempSpill(temp) => {
                    $spill_file_fn(temp.path(), max.min(logical_len), logical_len)
                }
            }
        }

        fn $as_f64_fn(backing: &$backing) -> Result<Vec<f64>, TetError> {
            match backing {
                $backing::InMemory(v) => Ok(v.iter().map(|&$v| $promote_inmem).collect()),
                $backing::TempSpill(temp) => {
                    let mmap = mmap_spill(temp.path())?;
                    Ok(bytemuck::cast_slice::<u8, $elem>(&mmap)
                        .iter()
                        .map(|&$s| $promote_spill)
                        .collect())
                }
            }
        }
    };
}

define_int_materialize! {
    i32;
    backing LogicalI32Backing;
    le_mod i32_le;
    scatter scatter_chunk_into_plan_i32;
    scatter_seq scatter_fill_sequential_i32;
    scatter_ty ScatterI32Fn;
    scatter_par materialize_scatter_fill_parallel_i32;
    core_fn materialize_read_plan_i32_le_core;
    read_fn materialize_read_plan_i32_le;
    spill_fn spill_read_plan_i32_le;
    type_label "i32";
    into_vec_fn materialize_into_vec_i32;
    spill_file_fn preview_from_spill_file_i32;
    preview_mat_fn preview_from_materialized_i32;
    as_f64_fn materialized_logical_as_f64_i32;
    promote_inmem |x| f64::from(x);
    promote_spill |x| f64::from(x);
}

define_int_materialize! {
    i64;
    backing LogicalI64Backing;
    le_mod i64_le;
    scatter scatter_chunk_into_plan_i64;
    scatter_seq scatter_fill_sequential_i64;
    scatter_ty ScatterI64Fn;
    scatter_par materialize_scatter_fill_parallel_i64;
    core_fn materialize_read_plan_i64_le_core;
    read_fn materialize_read_plan_i64_le;
    spill_fn spill_read_plan_i64_le;
    type_label "i64";
    into_vec_fn materialize_into_vec_i64;
    spill_file_fn preview_from_spill_file_i64;
    preview_mat_fn preview_from_materialized_i64;
    as_f64_fn materialized_logical_as_f64_i64;
    promote_inmem |x| x as f64;
    promote_spill |x| x as f64;
}

/// Spill a full logical `i32` or `i64` selection to `path` (dispatches by `dtype`).
///
/// # Errors
///
/// Returns [`TetError::Validation`] when `dtype` is not `i32` or `i64`, or on spill I/O failure.
pub fn spill_read_plan_int_le(
    mmap: &[u8],
    plan: &ReadPlan,
    path: &Path,
    dtype: ElementDtype,
) -> Result<u64, TetError> {
    match dtype {
        ElementDtype::I32 => spill_read_plan_i32_le(mmap, plan, path),
        ElementDtype::I64 => spill_read_plan_i64_le(mmap, plan, path),
        _ => Err(TetError::Validation(
            "spill_read_plan_int_le requires i32 or i64 dtype".into(),
        )),
    }
}

fn mmap_spill(path: &Path) -> Result<Mmap, TetError> {
    let file = std::fs::File::open(path)
        .map_err(|e| TetError::Validation(format!("temp spill read failed: {e}")))?;
    unsafe {
        Mmap::map(&file).map_err(|e| TetError::Validation(format!("temp spill mmap failed: {e}")))
    }
}

/// Load a materialized logical selection as `f64` for tier-C statistics.
pub(crate) fn materialized_logical_as_f64(
    materialized: &MaterializedLogical,
) -> Result<Vec<f64>, TetError> {
    match materialized {
        MaterializedLogical::F32 { backing, .. } => match backing {
            super::LogicalF32Backing::InMemory(v) => Ok(v.iter().map(|&x| f64::from(x)).collect()),
            super::LogicalF32Backing::TempSpill(temp) => {
                let mmap = mmap_spill(temp.path())?;
                Ok(bytemuck::cast_slice::<u8, f32>(&mmap)
                    .iter()
                    .map(|&x| f64::from(x))
                    .collect())
            }
        },
        MaterializedLogical::F64 { backing, .. } => match backing {
            super::LogicalF64Backing::InMemory(v) => Ok(v.clone()),
            super::LogicalF64Backing::TempSpill(temp) => {
                let mmap = mmap_spill(temp.path())?;
                Ok(bytemuck::cast_slice::<u8, f64>(&mmap).to_vec())
            }
        },
        MaterializedLogical::I32 { backing, .. } => materialized_logical_as_f64_i32(backing),
        MaterializedLogical::I64 { backing, .. } => materialized_logical_as_f64_i64(backing),
    }
}

#[derive(Copy, Clone)]
pub(crate) enum IntVisit {
    I32,
    I64,
}

impl IntVisit {
    pub(crate) fn visit_chunk_as_f64<F>(
        self,
        mmap: &[u8],
        plan: &ReadPlan,
        c: &PlannedChunkIo,
        visit: F,
    ) -> Result<u64, TetError>
    where
        F: FnMut(usize, f64) -> Result<(), TetError>,
    {
        match self {
            Self::I32 => visit_planned_chunk_i32_as_f64(mmap, plan, c, visit),
            Self::I64 => visit_planned_chunk_i64_as_f64(mmap, plan, c, visit),
        }
    }
}

macro_rules! int_scalar_fold_outcome {
    (
        i32: $preview:expr,
        max_preview: $max_preview:expr,
        n: $n:expr,
        total: $total:expr,
        operation: $operation:expr,
    ) => {
        build_fold_plan_outcome_typed(
            FoldPreviewBuffer::I32($preview),
            $max_preview,
            $n,
            $total,
            $operation,
        )
    };
    (
        i64: $preview:expr,
        max_preview: $max_preview:expr,
        n: $n:expr,
        total: $total:expr,
        operation: $operation:expr,
    ) => {
        build_fold_plan_outcome_typed(
            FoldPreviewBuffer::I64($preview),
            $max_preview,
            $n,
            $total,
            $operation,
        )
    };
}

macro_rules! int_scalar_fold_run {
    (
        elem $elem:ty;
        cast |$v:ident| $cast:expr;
        outcome i32;
        $mmap:ident;
        $plan:ident;
        $visit:expr;
        $preview_cap:expr;
        $max_preview:expr;
        $n:expr;
        $kind:ident;
        $acc:ident;
        on_value: $on_value:expr,
        finish => $finish:expr
    ) => {{
        let mut preview = vec![0 as $elem; $preview_cap];
        let mut total_bytes_read_from_disk: u64 = 0;
        let mut saw_preview = $preview_cap == 0;
        for c in &$plan.chunks {
            let chunk_bytes = $visit.visit_chunk_as_f64($mmap, $plan, c, |li, v| {
                $on_value(&mut $acc, li, v, $kind);
                if li < $preview_cap {
                    let $v = v;
                    preview[li] = $cast;
                    saw_preview = true;
                }
                Ok(())
            })?;
            accumulate_chunk_read_bytes(&mut total_bytes_read_from_disk, chunk_bytes)?;
        }
        if $acc.is_empty() {
            return Err(TetError::Validation(
                "operation requires at least one decoded value from the read plan".into(),
            ));
        }
        if $preview_cap > 0 && !saw_preview {
            return Err(TetError::Validation(
                "materialized selection has unset preview elements".into(),
            ));
        }
        let operation = $finish.into();
        Ok(int_scalar_fold_outcome!(
            i32: preview,
            max_preview: $max_preview,
            n: $n,
            total: total_bytes_read_from_disk,
            operation: operation,
        ))
    }};
    (
        elem $elem:ty;
        cast |$v:ident| $cast:expr;
        outcome i64;
        $mmap:ident;
        $plan:ident;
        $visit:expr;
        $preview_cap:expr;
        $max_preview:expr;
        $n:expr;
        $kind:ident;
        $acc:ident;
        on_value: $on_value:expr,
        finish => $finish:expr
    ) => {{
        let mut preview = vec![0 as $elem; $preview_cap];
        let mut total_bytes_read_from_disk: u64 = 0;
        let mut saw_preview = $preview_cap == 0;
        for c in &$plan.chunks {
            let chunk_bytes = $visit.visit_chunk_as_f64($mmap, $plan, c, |li, v| {
                $on_value(&mut $acc, li, v, $kind);
                if li < $preview_cap {
                    let $v = v;
                    preview[li] = $cast;
                    saw_preview = true;
                }
                Ok(())
            })?;
            accumulate_chunk_read_bytes(&mut total_bytes_read_from_disk, chunk_bytes)?;
        }
        if $acc.is_empty() {
            return Err(TetError::Validation(
                "operation requires at least one decoded value from the read plan".into(),
            ));
        }
        if $preview_cap > 0 && !saw_preview {
            return Err(TetError::Validation(
                "materialized selection has unset preview elements".into(),
            ));
        }
        let operation = $finish.into();
        Ok(int_scalar_fold_outcome!(
            i64: preview,
            max_preview: $max_preview,
            n: $n,
            total: total_bytes_read_from_disk,
            operation: operation,
        ))
    }};
}

fn int_scalar_fold_arg(
    mmap: &[u8],
    plan: &ReadPlan,
    max_preview: usize,
    kind: ReductionKind,
    visit: IntVisit,
    n: usize,
    preview_cap: usize,
) -> Result<FoldPlanOutcome, TetError> {
    match visit {
        IntVisit::I32 => {
            let mut acc = ArgIndexAccum::default();
            int_scalar_fold_run!(
                elem i32;
                cast |v| v as i32;
                outcome i32;
                mmap;
                plan;
                visit;
                preview_cap;
                max_preview;
                n;
                kind;
                acc;
                on_value: |acc: &mut ArgIndexAccum, li, v, kind| {
                    acc.push_f64(li as u64, v, kind);
                },
                finish => acc.finish_scalar(kind, n)
            )
        }
        IntVisit::I64 => {
            let mut acc = ArgIndexAccum::default();
            int_scalar_fold_run!(
                elem i64;
                cast |v| v as i64;
                outcome i64;
                mmap;
                plan;
                visit;
                preview_cap;
                max_preview;
                n;
                kind;
                acc;
                on_value: |acc: &mut ArgIndexAccum, li, v, kind| {
                    acc.push_f64(li as u64, v, kind);
                },
                finish => acc.finish_scalar(kind, n)
            )
        }
    }
}

fn int_scalar_fold_value(
    mmap: &[u8],
    plan: &ReadPlan,
    max_preview: usize,
    kind: ReductionKind,
    visit: IntVisit,
    n: usize,
    preview_cap: usize,
) -> Result<FoldPlanOutcome, TetError> {
    match visit {
        IntVisit::I32 => {
            let mut acc = ValueAccum::default();
            int_scalar_fold_run!(
                elem i32;
                cast |v| v as i32;
                outcome i32;
                mmap;
                plan;
                visit;
                preview_cap;
                max_preview;
                n;
                kind;
                acc;
                on_value: |acc: &mut ValueAccum, _li, v, _kind| acc.push_f64(v),
                finish => acc.finish_scalar(kind)
            )
        }
        IntVisit::I64 => {
            let mut acc = ValueAccum::default();
            int_scalar_fold_run!(
                elem i64;
                cast |v| v as i64;
                outcome i64;
                mmap;
                plan;
                visit;
                preview_cap;
                max_preview;
                n;
                kind;
                acc;
                on_value: |acc: &mut ValueAccum, _li, v, _kind| acc.push_f64(v),
                finish => acc.finish_scalar(kind)
            )
        }
    }
}

pub(crate) fn fold_read_plan_scalar_operation_int(
    mmap: &[u8],
    plan: &ReadPlan,
    max_preview: usize,
    kind: ReductionKind,
    dtype: ElementDtype,
) -> Result<FoldPlanOutcome, TetError> {
    if crate::query::fold::parallel_fold::use_parallel_fold(plan) {
        return crate::query::fold::parallel_fold::fold_read_plan_scalar_operation_int_parallel(
            mmap,
            plan,
            max_preview,
            kind,
            dtype,
        );
    }
    let visit = match dtype {
        ElementDtype::I32 => IntVisit::I32,
        ElementDtype::I64 => IntVisit::I64,
        _ => {
            return Err(TetError::Validation(
                "integer fold requires i32 or i64 dtype".into(),
            ));
        }
    };
    let n = plan.logical_f32_element_count;
    let preview_cap = max_preview.min(n);
    match kind {
        ReductionKind::ArgMin | ReductionKind::ArgMax => {
            int_scalar_fold_arg(mmap, plan, max_preview, kind, visit, n, preview_cap)
        }
        _ => int_scalar_fold_value(mmap, plan, max_preview, kind, visit, n, preview_cap),
    }
}
