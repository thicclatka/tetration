//! Parallel chunk-streaming scalar and partial-axis folds (Rayon).

use rayon::prelude::*;

use crate::query::decode::chunk_decode::{visit_planned_chunk, visit_planned_chunk_f64};
use crate::query::decode::indexing::coords_from_linear_row_major;
use crate::query::fold::partial_fold::{
    partial_arg_fields, partial_fields, validate_fold_preview_f64,
};
use crate::query::fold::partial_geometry::{
    PartialAxisLayout, fiber_linear_index, partial_axis_layout, reduced_index,
};
use crate::query::fold::reduction::{ArgIndexAccum, ReductionKind, ValueAccum};
use crate::query::fold::shared::{FoldPlanOutcome, build_fold_plan_outcome, validate_fold_preview};
use crate::query::materialize::int::IntVisit;
use crate::query::types::{OperationPreviewFields, ReadPlan, TetError};
use crate::utils::dtype::ElementDtype;

/// Use Rayon when more than one planned chunk is touched (matches materialize dispatch).
#[must_use]
pub(crate) fn use_parallel_fold(plan: &ReadPlan) -> bool {
    plan.chunks.len() > 1
}

struct ScalarChunkWork {
    bytes: u64,
    value: ValueAccum,
    arg: ArgIndexAccum,
}

struct PartialChunkValue {
    bytes: u64,
    cells: Vec<ValueAccum>,
    saw_any: bool,
}

struct PartialChunkArg {
    bytes: u64,
    cells: Vec<ArgIndexAccum>,
    saw_any: bool,
}

fn sum_chunk_bytes(bytes: impl IntoIterator<Item = u64>) -> Result<u64, TetError> {
    bytes.into_iter().try_fold(0u64, |a, b| {
        a.checked_add(b)
            .ok_or_else(|| TetError::Validation("total bytes read overflow".into()))
    })
}

fn merge_scalar_chunks(
    parts: &[ScalarChunkWork],
    kind: ReductionKind,
    n: usize,
) -> Result<OperationPreviewFields, TetError> {
    match kind {
        ReductionKind::ArgMin | ReductionKind::ArgMax => {
            let mut acc = ArgIndexAccum::default();
            for p in parts {
                acc.merge_from(&p.arg, kind);
            }
            if acc.is_empty() {
                return Err(TetError::Validation(
                    "operation requires at least one decoded value from the read plan".into(),
                ));
            }
            Ok(acc.finish_scalar(kind, n).into())
        }
        _ => {
            let mut acc = ValueAccum::default();
            for p in parts {
                acc.merge_from(&p.value);
            }
            if acc.is_empty() {
                return Err(TetError::Validation(
                    "operation requires at least one decoded value from the read plan".into(),
                ));
            }
            Ok(acc.finish_scalar(kind).into())
        }
    }
}

fn merge_partial_value_cells(dst: &mut [ValueAccum], src: &[ValueAccum]) {
    for (d, s) in dst.iter_mut().zip(src) {
        d.merge_from(s);
    }
}

fn merge_partial_arg_cells(dst: &mut [ArgIndexAccum], src: &[ArgIndexAccum], kind: ReductionKind) {
    for (d, s) in dst.iter_mut().zip(src) {
        d.merge_from(s, kind);
    }
}

fn reduced_cell_index(
    li: usize,
    shape: &[u64],
    layout: &PartialAxisLayout,
) -> Result<(usize, u64), TetError> {
    let coords = coords_from_linear_row_major(li, shape)?;
    let oi = reduced_index(&coords, &layout.axis_set, &layout.out_shape)?;
    let fi = fiber_linear_index(&coords, &layout.axis_indices, shape)? as u64;
    Ok((oi, fi))
}

fn write_disjoint_preview_f32(preview_addr: usize, preview_len: usize, li: usize, v: f32) {
    if li < preview_len {
        // SAFETY: planned chunks write disjoint logical indices.
        let preview =
            unsafe { std::slice::from_raw_parts_mut(preview_addr as *mut f32, preview_len) };
        preview[li] = v;
    }
}

fn write_disjoint_preview_f64(preview_addr: usize, preview_len: usize, li: usize, v: f64) {
    if li < preview_len {
        let preview =
            unsafe { std::slice::from_raw_parts_mut(preview_addr as *mut f64, preview_len) };
        preview[li] = v;
    }
}

fn write_disjoint_preview_i32(preview_addr: usize, preview_len: usize, li: usize, v: f64) {
    if li < preview_len {
        let preview =
            unsafe { std::slice::from_raw_parts_mut(preview_addr as *mut i32, preview_len) };
        preview[li] = v as i32;
    }
}

fn write_disjoint_preview_i64(preview_addr: usize, preview_len: usize, li: usize, v: f64) {
    if li < preview_len {
        let preview =
            unsafe { std::slice::from_raw_parts_mut(preview_addr as *mut i64, preview_len) };
        preview[li] = v as i64;
    }
}

#[derive(Copy, Clone)]
enum ScalarFoldVisit {
    F32,
    F64,
    Int(IntVisit),
}

impl ScalarFoldVisit {
    fn visit_chunk<F>(
        self,
        mmap: &[u8],
        plan: &ReadPlan,
        c: &crate::query::types::PlannedChunkIo,
        mut visit: F,
    ) -> Result<u64, TetError>
    where
        F: FnMut(usize, f64) -> Result<(), TetError>,
    {
        match self {
            Self::F32 => visit_planned_chunk(mmap, plan, c, |li, v| visit(li, f64::from(v))),
            Self::F64 => visit_planned_chunk_f64(mmap, plan, c, visit),
            Self::Int(v) => v.visit_chunk_as_f64(mmap, plan, c, visit),
        }
    }

    fn push_value(self, acc: &mut ValueAccum, v: f64) {
        match self {
            Self::F32 => acc.push(v as f32),
            Self::F64 | Self::Int(_) => acc.push_f64(v),
        }
    }

    fn push_arg(self, acc: &mut ArgIndexAccum, li: usize, v: f64, kind: ReductionKind) {
        match self {
            Self::F32 => acc.push(li as u64, v as f32, kind),
            Self::F64 | Self::Int(_) => acc.push_f64(li as u64, v, kind),
        }
    }

    fn write_preview(self, preview_addr: usize, preview_len: usize, li: usize, v: f64) {
        match self {
            Self::F32 => write_disjoint_preview_f32(preview_addr, preview_len, li, v as f32),
            Self::F64 => write_disjoint_preview_f64(preview_addr, preview_len, li, v),
            Self::Int(IntVisit::I32) => write_disjoint_preview_i32(preview_addr, preview_len, li, v),
            Self::Int(IntVisit::I64) => write_disjoint_preview_i64(preview_addr, preview_len, li, v),
        }
    }
}

fn fold_read_plan_scalar_parallel(
    mmap: &[u8],
    plan: &ReadPlan,
    max_preview: usize,
    kind: ReductionKind,
    visit: ScalarFoldVisit,
) -> Result<FoldPlanOutcome, TetError> {
    let n = plan.logical_f32_element_count;
    let preview_cap = max_preview.min(n);
    let mut f32_preview = vec![f32::NAN; preview_cap];
    let mut f64_preview = vec![f64::NAN; preview_cap];
    let mut i32_preview = vec![0i32; preview_cap];
    let mut i64_preview = vec![0i64; preview_cap];
    let (preview_addr, preview_len) = match visit {
        ScalarFoldVisit::F32 => (f32_preview.as_mut_ptr() as usize, f32_preview.len()),
        ScalarFoldVisit::F64 => (f64_preview.as_mut_ptr() as usize, f64_preview.len()),
        ScalarFoldVisit::Int(IntVisit::I32) => (i32_preview.as_mut_ptr() as usize, i32_preview.len()),
        ScalarFoldVisit::Int(IntVisit::I64) => (i64_preview.as_mut_ptr() as usize, i64_preview.len()),
    };

    let parts: Vec<ScalarChunkWork> = plan
        .chunks
        .par_iter()
        .map(|c| {
            let mut value = ValueAccum::default();
            let mut arg = ArgIndexAccum::default();
            let bytes = visit.visit_chunk(mmap, plan, c, |li, v| {
                match kind {
                    ReductionKind::ArgMin | ReductionKind::ArgMax => {
                        visit.push_arg(&mut arg, li, v, kind);
                    }
                    _ => visit.push_value(&mut value, v),
                }
                visit.write_preview(preview_addr, preview_len, li, v);
                Ok(())
            })?;
            Ok(ScalarChunkWork { bytes, value, arg })
        })
        .collect::<Result<Vec<_>, TetError>>()?;

    let total_bytes_read_from_disk = sum_chunk_bytes(parts.iter().map(|p| p.bytes))?;
    let operation = merge_scalar_chunks(&parts, kind, n)?;

    match visit {
        ScalarFoldVisit::F32 => {
            validate_fold_preview(true, &f32_preview, preview_cap)?;
            Ok(build_fold_plan_outcome(
                f32_preview,
                max_preview,
                n,
                total_bytes_read_from_disk,
                operation,
            ))
        }
        ScalarFoldVisit::F64 => {
            validate_fold_preview_f64(true, &f64_preview, preview_cap)?;
            Ok(FoldPlanOutcome {
                f32_preview: Vec::new(),
                f64_preview: if max_preview == 0 {
                    Vec::new()
                } else {
                    f64_preview
                },
                i32_preview: Vec::new(),
                i64_preview: Vec::new(),
                f32_preview_truncated: false,
                f64_preview_truncated: n > max_preview,
                i32_preview_truncated: false,
                i64_preview_truncated: false,
                total_bytes_read_from_disk,
                operation,
            })
        }
        ScalarFoldVisit::Int(IntVisit::I32) => Ok(FoldPlanOutcome {
            f32_preview: Vec::new(),
            f64_preview: Vec::new(),
            i32_preview: if max_preview == 0 {
                Vec::new()
            } else {
                i32_preview
            },
            i64_preview: Vec::new(),
            f32_preview_truncated: false,
            f64_preview_truncated: false,
            i32_preview_truncated: n > max_preview,
            i64_preview_truncated: false,
            total_bytes_read_from_disk,
            operation,
        }),
        ScalarFoldVisit::Int(IntVisit::I64) => Ok(FoldPlanOutcome {
            f32_preview: Vec::new(),
            f64_preview: Vec::new(),
            i32_preview: Vec::new(),
            i64_preview: if max_preview == 0 {
                Vec::new()
            } else {
                i64_preview
            },
            f32_preview_truncated: false,
            f64_preview_truncated: false,
            i32_preview_truncated: false,
            i64_preview_truncated: n > max_preview,
            total_bytes_read_from_disk,
            operation,
        }),
    }
}

/// Parallel scalar fold over `f32` planned chunks.
pub(crate) fn fold_read_plan_scalar_operation_parallel(
    mmap: &[u8],
    plan: &ReadPlan,
    max_f32: usize,
    kind: ReductionKind,
) -> Result<FoldPlanOutcome, TetError> {
    fold_read_plan_scalar_parallel(mmap, plan, max_f32, kind, ScalarFoldVisit::F32)
}

/// Parallel scalar fold over `f64` planned chunks.
pub(crate) fn fold_read_plan_scalar_operation_f64_parallel(
    mmap: &[u8],
    plan: &ReadPlan,
    max_preview: usize,
    kind: ReductionKind,
) -> Result<FoldPlanOutcome, TetError> {
    fold_read_plan_scalar_parallel(mmap, plan, max_preview, kind, ScalarFoldVisit::F64)
}

/// Parallel scalar fold for `i32` / `i64` (promoted to `f64` accumulators).
pub(crate) fn fold_read_plan_scalar_operation_int_parallel(
    mmap: &[u8],
    plan: &ReadPlan,
    max_preview: usize,
    kind: ReductionKind,
    dtype: ElementDtype,
) -> Result<FoldPlanOutcome, TetError> {
    let visit = match dtype {
        ElementDtype::I32 => ScalarFoldVisit::Int(IntVisit::I32),
        ElementDtype::I64 => ScalarFoldVisit::Int(IntVisit::I64),
        _ => {
            return Err(TetError::Validation(
                "integer fold requires i32 or i64 dtype".into(),
            ));
        }
    };
    fold_read_plan_scalar_parallel(mmap, plan, max_preview, kind, visit)
}

#[derive(Copy, Clone)]
enum PartialFoldVisit {
    F32,
    F64,
}

impl PartialFoldVisit {
    fn visit_chunk<F>(
        self,
        mmap: &[u8],
        plan: &ReadPlan,
        c: &crate::query::types::PlannedChunkIo,
        mut visit: F,
    ) -> Result<u64, TetError>
    where
        F: FnMut(usize, f64) -> Result<(), TetError>,
    {
        match self {
            Self::F32 => visit_planned_chunk(mmap, plan, c, |li, v| visit(li, f64::from(v))),
            Self::F64 => visit_planned_chunk_f64(mmap, plan, c, visit),
        }
    }

    fn push_value(self, cell: &mut ValueAccum, v: f64) {
        match self {
            Self::F32 => cell.push(v as f32),
            Self::F64 => cell.push_f64(v),
        }
    }

    fn push_arg(self, cell: &mut ArgIndexAccum, fi: u64, v: f64, kind: ReductionKind) {
        match self {
            Self::F32 => cell.push(fi, v as f32, kind),
            Self::F64 => cell.push_f64(fi, v, kind),
        }
    }

    fn write_preview(self, preview_addr: usize, preview_len: usize, li: usize, v: f64) {
        match self {
            Self::F32 => write_disjoint_preview_f32(preview_addr, preview_len, li, v as f32),
            Self::F64 => write_disjoint_preview_f64(preview_addr, preview_len, li, v),
        }
    }

    fn validate_preview(
        self,
        saw_any: bool,
        f32_preview: &[f32],
        f64_preview: &[f64],
        preview_cap: usize,
    ) -> Result<(), TetError> {
        match self {
            Self::F32 => validate_fold_preview(saw_any, f32_preview, preview_cap),
            Self::F64 => validate_fold_preview_f64(saw_any, f64_preview, preview_cap),
        }
    }
}

struct PartialParallelWork {
    operation: OperationPreviewFields,
    total_bytes: u64,
    saw_any: bool,
}

struct PartialParallelCtx<'a> {
    mmap: &'a [u8],
    plan: &'a ReadPlan,
    visit: PartialFoldVisit,
    kind: ReductionKind,
    layout: &'a PartialAxisLayout,
    shape: &'a [u64],
    preview_addr: usize,
    preview_len: usize,
    n: usize,
}

fn parallel_partial_arg(ctx: &PartialParallelCtx<'_>) -> Result<PartialParallelWork, TetError> {
    let out_len = ctx.layout.out_len;
    let parts: Vec<PartialChunkArg> = ctx
        .plan
        .chunks
        .par_iter()
        .map(|c| {
            let mut cells = vec![ArgIndexAccum::default(); out_len];
            let mut saw_any = false;
            let bytes = ctx.visit.visit_chunk(ctx.mmap, ctx.plan, c, |li, v| {
                saw_any = true;
                ctx.visit
                    .write_preview(ctx.preview_addr, ctx.preview_len, li, v);
                let (oi, fi) = reduced_cell_index(li, ctx.shape, ctx.layout)?;
                ctx.visit.push_arg(&mut cells[oi], fi, v, ctx.kind);
                Ok(())
            })?;
            Ok(PartialChunkArg {
                bytes,
                cells,
                saw_any,
            })
        })
        .collect::<Result<Vec<_>, TetError>>()?;

    let mut merged = vec![ArgIndexAccum::default(); out_len];
    let mut saw_any = false;
    let mut total_bytes = 0u64;
    for p in &parts {
        saw_any |= p.saw_any;
        merge_partial_arg_cells(&mut merged, &p.cells, ctx.kind);
        total_bytes = total_bytes
            .checked_add(p.bytes)
            .ok_or_else(|| TetError::Validation("total bytes read overflow".into()))?;
    }
    Ok(PartialParallelWork {
        operation: partial_arg_fields(ctx.kind, ctx.n, &ctx.layout.out_shape, &merged),
        total_bytes,
        saw_any,
    })
}

fn parallel_partial_value(ctx: &PartialParallelCtx<'_>) -> Result<PartialParallelWork, TetError> {
    let out_len = ctx.layout.out_len;
    let parts: Vec<PartialChunkValue> = ctx
        .plan
        .chunks
        .par_iter()
        .map(|c| {
            let mut cells = vec![ValueAccum::default(); out_len];
            let mut saw_any = false;
            let bytes = ctx.visit.visit_chunk(ctx.mmap, ctx.plan, c, |li, v| {
                saw_any = true;
                ctx.visit
                    .write_preview(ctx.preview_addr, ctx.preview_len, li, v);
                let (oi, _) = reduced_cell_index(li, ctx.shape, ctx.layout)?;
                ctx.visit.push_value(&mut cells[oi], v);
                Ok(())
            })?;
            Ok(PartialChunkValue {
                bytes,
                cells,
                saw_any,
            })
        })
        .collect::<Result<Vec<_>, TetError>>()?;

    let mut merged = vec![ValueAccum::default(); out_len];
    let mut saw_any = false;
    let mut total_bytes = 0u64;
    for p in &parts {
        saw_any |= p.saw_any;
        merge_partial_value_cells(&mut merged, &p.cells);
        total_bytes = total_bytes
            .checked_add(p.bytes)
            .ok_or_else(|| TetError::Validation("total bytes read overflow".into()))?;
    }
    let reduced: Vec<f64> = merged.iter().map(|c| c.finish_f64(ctx.kind)).collect();
    Ok(PartialParallelWork {
        operation: partial_fields(ctx.kind, ctx.n, &ctx.layout.out_shape, &reduced, &merged),
        total_bytes,
        saw_any,
    })
}

fn parallel_partial_axis_fold(
    mmap: &[u8],
    plan: &ReadPlan,
    max_preview: usize,
    kind: ReductionKind,
    axis_labels: &[String],
    visit: PartialFoldVisit,
) -> Result<FoldPlanOutcome, TetError> {
    let layout = partial_axis_layout(plan, axis_labels)?;
    let shape = plan.logical_selection_shape.clone();
    let n = plan.logical_f32_element_count;
    let preview_cap = max_preview.min(n);
    let mut f32_preview = vec![f32::NAN; preview_cap];
    let mut f64_preview = vec![f64::NAN; preview_cap];
    let (preview_addr, preview_len) = match visit {
        PartialFoldVisit::F32 => (f32_preview.as_mut_ptr() as usize, f32_preview.len()),
        PartialFoldVisit::F64 => (f64_preview.as_mut_ptr() as usize, f64_preview.len()),
    };

    let ctx = PartialParallelCtx {
        mmap,
        plan,
        visit,
        kind,
        layout: &layout,
        shape: &shape,
        preview_addr,
        preview_len,
        n,
    };
    let work = match kind {
        ReductionKind::ArgMin | ReductionKind::ArgMax => parallel_partial_arg(&ctx)?,
        _ => parallel_partial_value(&ctx)?,
    };

    visit.validate_preview(work.saw_any, &f32_preview, &f64_preview, preview_cap)?;

    match visit {
        PartialFoldVisit::F32 => Ok(build_fold_plan_outcome(
            f32_preview,
            max_preview,
            n,
            work.total_bytes,
            work.operation,
        )),
        PartialFoldVisit::F64 => Ok(FoldPlanOutcome {
            f32_preview: Vec::new(),
            f64_preview: if max_preview == 0 {
                Vec::new()
            } else {
                f64_preview
            },
            i32_preview: Vec::new(),
            i64_preview: Vec::new(),
            f32_preview_truncated: false,
            f64_preview_truncated: n > max_preview,
            i32_preview_truncated: false,
            i64_preview_truncated: false,
            total_bytes_read_from_disk: work.total_bytes,
            operation: work.operation,
        }),
    }
}

/// Parallel partial-axis fold (`f32`).
pub(crate) fn fold_read_plan_partial_operation_parallel(
    mmap: &[u8],
    plan: &ReadPlan,
    max_preview: usize,
    kind: ReductionKind,
    axis_labels: &[String],
) -> Result<FoldPlanOutcome, TetError> {
    parallel_partial_axis_fold(mmap, plan, max_preview, kind, axis_labels, PartialFoldVisit::F32)
}

/// Parallel partial-axis fold (`f64`).
pub(crate) fn fold_read_plan_partial_operation_f64_parallel(
    mmap: &[u8],
    plan: &ReadPlan,
    max_preview: usize,
    kind: ReductionKind,
    axis_labels: &[String],
) -> Result<FoldPlanOutcome, TetError> {
    parallel_partial_axis_fold(mmap, plan, max_preview, kind, axis_labels, PartialFoldVisit::F64)
}