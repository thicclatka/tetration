//! Integer scalar fold (sequential path).

use crate::query::dispatch::accumulate_chunk_read_bytes;
use crate::query::types::{PlannedChunkIo, ReadPlan, TetError};
use crate::utils::dtype::ElementDtype;

use crate::query::decode::chunk_decode::{
    visit_planned_chunk_i16_as_f64, visit_planned_chunk_i32_as_f64, visit_planned_chunk_i64_as_f64,
    visit_planned_chunk_u8_as_f64, visit_planned_chunk_u16_as_f64,
};
use crate::query::fold::reduction::{ArgIndexAccum, ReductionKind, ValueAccum};
use crate::query::fold::shared::{
    FoldPlanOutcome, FoldPreviewBuffer, build_fold_plan_outcome_typed,
};

#[derive(Copy, Clone)]
pub(crate) enum IntVisit {
    I32,
    I64,
    U8,
    U16,
    I16,
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
            Self::U8 => visit_planned_chunk_u8_as_f64(mmap, plan, c, visit),
            Self::U16 => visit_planned_chunk_u16_as_f64(mmap, plan, c, visit),
            Self::I16 => visit_planned_chunk_i16_as_f64(mmap, plan, c, visit),
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
    (
        u8: $preview:expr,
        max_preview: $max_preview:expr,
        n: $n:expr,
        total: $total:expr,
        operation: $operation:expr,
    ) => {
        build_fold_plan_outcome_typed(
            FoldPreviewBuffer::U8($preview),
            $max_preview,
            $n,
            $total,
            $operation,
        )
    };
    (
        u16: $preview:expr,
        max_preview: $max_preview:expr,
        n: $n:expr,
        total: $total:expr,
        operation: $operation:expr,
    ) => {
        build_fold_plan_outcome_typed(
            FoldPreviewBuffer::U16($preview),
            $max_preview,
            $n,
            $total,
            $operation,
        )
    };
    (
        i16: $preview:expr,
        max_preview: $max_preview:expr,
        n: $n:expr,
        total: $total:expr,
        operation: $operation:expr,
    ) => {
        build_fold_plan_outcome_typed(
            FoldPreviewBuffer::I16($preview),
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
        $sequential_io:expr;
        on_value: $on_value:expr,
        finish => $finish:expr
    ) => {{
        let mut preview = vec![0 as $elem; $preview_cap];
        let mut total_bytes_read_from_disk: u64 = 0;
        let mut saw_preview = $preview_cap == 0;
        for i in crate::query::fold::fold_policy::chunk_indices_for_fold($plan, $sequential_io) {
            let c = &$plan.chunks[i];
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
        $sequential_io:expr;
        on_value: $on_value:expr,
        finish => $finish:expr
    ) => {{
        let mut preview = vec![0 as $elem; $preview_cap];
        let mut total_bytes_read_from_disk: u64 = 0;
        let mut saw_preview = $preview_cap == 0;
        for i in crate::query::fold::fold_policy::chunk_indices_for_fold($plan, $sequential_io) {
            let c = &$plan.chunks[i];
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
    (
        elem $elem:ty;
        cast |$v:ident| $cast:expr;
        outcome u8;
        $mmap:ident;
        $plan:ident;
        $visit:expr;
        $preview_cap:expr;
        $max_preview:expr;
        $n:expr;
        $kind:ident;
        $acc:ident;
        $sequential_io:expr;
        on_value: $on_value:expr,
        finish => $finish:expr
    ) => {{
        let mut preview = vec![0 as $elem; $preview_cap];
        let mut total_bytes_read_from_disk: u64 = 0;
        let mut saw_preview = $preview_cap == 0;
        for i in crate::query::fold::fold_policy::chunk_indices_for_fold($plan, $sequential_io) {
            let c = &$plan.chunks[i];
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
            u8: preview,
            max_preview: $max_preview,
            n: $n,
            total: total_bytes_read_from_disk,
            operation: operation,
        ))
    }};
    (
        elem $elem:ty;
        cast |$v:ident| $cast:expr;
        outcome u16;
        $mmap:ident;
        $plan:ident;
        $visit:expr;
        $preview_cap:expr;
        $max_preview:expr;
        $n:expr;
        $kind:ident;
        $acc:ident;
        $sequential_io:expr;
        on_value: $on_value:expr,
        finish => $finish:expr
    ) => {{
        let mut preview = vec![0 as $elem; $preview_cap];
        let mut total_bytes_read_from_disk: u64 = 0;
        let mut saw_preview = $preview_cap == 0;
        for i in crate::query::fold::fold_policy::chunk_indices_for_fold($plan, $sequential_io) {
            let c = &$plan.chunks[i];
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
            u16: preview,
            max_preview: $max_preview,
            n: $n,
            total: total_bytes_read_from_disk,
            operation: operation,
        ))
    }};
    (
        elem $elem:ty;
        cast |$v:ident| $cast:expr;
        outcome i16;
        $mmap:ident;
        $plan:ident;
        $visit:expr;
        $preview_cap:expr;
        $max_preview:expr;
        $n:expr;
        $kind:ident;
        $acc:ident;
        $sequential_io:expr;
        on_value: $on_value:expr,
        finish => $finish:expr
    ) => {{
        let mut preview = vec![0 as $elem; $preview_cap];
        let mut total_bytes_read_from_disk: u64 = 0;
        let mut saw_preview = $preview_cap == 0;
        for i in crate::query::fold::fold_policy::chunk_indices_for_fold($plan, $sequential_io) {
            let c = &$plan.chunks[i];
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
            i16: preview,
            max_preview: $max_preview,
            n: $n,
            total: total_bytes_read_from_disk,
            operation: operation,
        ))
    }};
}

/// Shared inputs for sequential integer scalar fold (`i32` / `i64` promoted to `f64`).
struct IntScalarFoldCtx<'a> {
    mmap: &'a [u8],
    plan: &'a ReadPlan,
    max_preview: usize,
    kind: ReductionKind,
    visit: IntVisit,
    n: usize,
    preview_cap: usize,
    sequential_io: bool,
}

fn int_scalar_fold_arg(ctx: &IntScalarFoldCtx<'_>) -> Result<FoldPlanOutcome, TetError> {
    let IntScalarFoldCtx {
        mmap,
        plan,
        max_preview,
        kind,
        visit,
        n,
        preview_cap,
        sequential_io,
    } = *ctx;
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
                sequential_io;
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
                sequential_io;
                on_value: |acc: &mut ArgIndexAccum, li, v, kind| {
                    acc.push_f64(li as u64, v, kind);
                },
                finish => acc.finish_scalar(kind, n)
            )
        }
        IntVisit::U8 => {
            let mut acc = ArgIndexAccum::default();
            int_scalar_fold_run!(
                elem u8;
                cast |v| v as u8;
                outcome u8;
                mmap;
                plan;
                visit;
                preview_cap;
                max_preview;
                n;
                kind;
                acc;
                sequential_io;
                on_value: |acc: &mut ArgIndexAccum, li, v, kind| {
                    acc.push_f64(li as u64, v, kind);
                },
                finish => acc.finish_scalar(kind, n)
            )
        }
        IntVisit::U16 => {
            let mut acc = ArgIndexAccum::default();
            int_scalar_fold_run!(
                elem u16;
                cast |v| v as u16;
                outcome u16;
                mmap;
                plan;
                visit;
                preview_cap;
                max_preview;
                n;
                kind;
                acc;
                sequential_io;
                on_value: |acc: &mut ArgIndexAccum, li, v, kind| {
                    acc.push_f64(li as u64, v, kind);
                },
                finish => acc.finish_scalar(kind, n)
            )
        }
        IntVisit::I16 => {
            let mut acc = ArgIndexAccum::default();
            int_scalar_fold_run!(
                elem i16;
                cast |v| v as i16;
                outcome i16;
                mmap;
                plan;
                visit;
                preview_cap;
                max_preview;
                n;
                kind;
                acc;
                sequential_io;
                on_value: |acc: &mut ArgIndexAccum, li, v, kind| {
                    acc.push_f64(li as u64, v, kind);
                },
                finish => acc.finish_scalar(kind, n)
            )
        }
    }
}

fn int_scalar_fold_value(ctx: &IntScalarFoldCtx<'_>) -> Result<FoldPlanOutcome, TetError> {
    let IntScalarFoldCtx {
        mmap,
        plan,
        max_preview,
        kind,
        visit,
        n,
        preview_cap,
        sequential_io,
    } = *ctx;
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
                sequential_io;
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
                sequential_io;
                on_value: |acc: &mut ValueAccum, _li, v, _kind| acc.push_f64(v),
                finish => acc.finish_scalar(kind)
            )
        }
        IntVisit::U8 => {
            let mut acc = ValueAccum::default();
            int_scalar_fold_run!(
                elem u8;
                cast |v| v as u8;
                outcome u8;
                mmap;
                plan;
                visit;
                preview_cap;
                max_preview;
                n;
                kind;
                acc;
                sequential_io;
                on_value: |acc: &mut ValueAccum, _li, v, _kind| acc.push_f64(v),
                finish => acc.finish_scalar(kind)
            )
        }
        IntVisit::U16 => {
            let mut acc = ValueAccum::default();
            int_scalar_fold_run!(
                elem u16;
                cast |v| v as u16;
                outcome u16;
                mmap;
                plan;
                visit;
                preview_cap;
                max_preview;
                n;
                kind;
                acc;
                sequential_io;
                on_value: |acc: &mut ValueAccum, _li, v, _kind| acc.push_f64(v),
                finish => acc.finish_scalar(kind)
            )
        }
        IntVisit::I16 => {
            let mut acc = ValueAccum::default();
            int_scalar_fold_run!(
                elem i16;
                cast |v| v as i16;
                outcome i16;
                mmap;
                plan;
                visit;
                preview_cap;
                max_preview;
                n;
                kind;
                acc;
                sequential_io;
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
    policy: &crate::query::fold::fold_policy::FoldIoPolicy,
) -> Result<FoldPlanOutcome, TetError> {
    let visit = match dtype {
        ElementDtype::I32 => IntVisit::I32,
        ElementDtype::I64 => IntVisit::I64,
        ElementDtype::U8 => IntVisit::U8,
        ElementDtype::U16 => IntVisit::U16,
        ElementDtype::I16 => IntVisit::I16,
        _ => {
            return Err(TetError::Validation(
                "integer fold requires i32, i64, u8, u16, or i16 dtype".into(),
            ));
        }
    };
    if crate::query::fold::parallel_fold::use_parallel_fold(plan, policy) {
        return crate::query::fold::parallel_fold::fold_read_plan_scalar_operation_int_parallel(
            mmap,
            plan,
            max_preview,
            kind,
            dtype,
            policy.fold_workers,
        );
    }
    let n = plan.logical_f32_element_count;
    let preview_cap = max_preview.min(n);
    let ctx = IntScalarFoldCtx {
        mmap,
        plan,
        max_preview,
        kind,
        visit,
        n,
        preview_cap,
        sequential_io: policy.sequential_io,
    };
    match kind {
        ReductionKind::ArgMin | ReductionKind::ArgMax => int_scalar_fold_arg(&ctx),
        _ => int_scalar_fold_value(&ctx),
    }
}
