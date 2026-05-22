use crate::catalog::DTYPE_F32;
use crate::query::types::{
    Operation, OperationPreviewFields, OutputHint, OutputHints, QueryExecutionPreview, ReadPlan,
    TetError,
};

use super::budget::{ExecutionBudget, MemoryStrategy};
use super::materialize::{
    FoldPlanOutcome, fold_read_plan_scalar_operation, materialize_read_plan_f32_le,
    spill_read_plan_f32_le,
};
use super::parallel::materialize_read_plan_f32_le_parallel;
use super::partial_fold::fold_read_plan_partial_operation;
use super::reduction::ReductionKind;

fn scalar_reduction_kind(op: &Operation) -> Option<ReductionKind> {
    op.axes().is_empty().then(|| ReductionKind::from(op))
}

fn materialize_read_plan_f32_le_for_execution(
    mmap: &[u8],
    plan: &ReadPlan,
    max_elements: Option<usize>,
) -> Result<(Vec<f32>, bool, u64), TetError> {
    if plan.chunks.len() <= 1 {
        materialize_read_plan_f32_le(mmap, plan, max_elements)
    } else {
        materialize_read_plan_f32_le_parallel(mmap, plan, max_elements)
    }
}

fn spill_requested(output: Option<&OutputHints>) -> Option<&str> {
    match output.and_then(|o| o.preferred.as_ref()) {
        Some(OutputHint::SpillArray { handle }) => Some(handle.as_str()),
        _ => None,
    }
}

fn fold_outcome_to_preview(
    folded: FoldPlanOutcome,
    strategy: MemoryStrategy,
    budget: ExecutionBudget,
    plan: &ReadPlan,
) -> QueryExecutionPreview {
    let mut preview = QueryExecutionPreview::with_operation_and_io(
        folded.total_bytes_read_from_disk,
        folded.f32_preview,
        folded.f32_preview_truncated,
        folded.operation,
        Some(strategy.as_str()),
        None,
        None,
    );
    attach_budget_fields(&mut preview, budget, plan);
    preview
}

fn attach_budget_fields(
    preview: &mut QueryExecutionPreview,
    budget: ExecutionBudget,
    plan: &ReadPlan,
) {
    preview.memory_budget_bytes = Some(budget.memory_budget_bytes);
    preview.host_available_ram_bytes = budget.host_available_ram_bytes;
    preview.logical_selection_f32_bytes = budget
        .logical_f32_bytes(plan.logical_f32_element_count)
        .ok();
}

pub(super) fn build_execution_preview(
    mmap: &[u8],
    plan: &ReadPlan,
    dtype: u32,
    operation: Option<&Operation>,
    output: Option<&OutputHints>,
    max_f32: usize,
    budget: ExecutionBudget,
) -> Result<QueryExecutionPreview, TetError> {
    if dtype != DTYPE_F32 {
        return Err(TetError::Validation(
            "f32 preview requires dataset dtype f32 (DTYPE_F32 = 1)".into(),
        ));
    }

    match operation {
        None => {
            if let Some(spill_path) = spill_requested(output) {
                let path = std::path::Path::new(spill_path);
                let total_bytes_read_from_disk = spill_read_plan_f32_le(mmap, plan, path)?;
                let (f32_preview, f32_preview_truncated, _) =
                    materialize_read_plan_f32_le_for_execution(mmap, plan, Some(max_f32))?;
                let spill_bytes = budget.logical_f32_bytes(plan.logical_f32_element_count)?;
                let mut preview = QueryExecutionPreview::with_operation_and_io(
                    total_bytes_read_from_disk,
                    f32_preview,
                    f32_preview_truncated,
                    OperationPreviewFields::default(),
                    Some(MemoryStrategy::MmapSpill.as_str()),
                    Some(spill_path.to_string()),
                    Some(spill_bytes),
                );
                attach_budget_fields(&mut preview, budget, plan);
                return Ok(preview);
            }
            if budget.full_tensor_exceeds_budget(plan)? && max_f32 == 0 {
                return Err(TetError::Validation(format!(
                    "logical selection ({} f32, {} bytes) exceeds memory_budget_bytes ({}); \
                     use a positive preview cap, an `operation`, output spill, or raise execution.memory_budget_bytes / memory_budget_percent_bps",
                    plan.logical_f32_element_count,
                    budget.logical_f32_bytes(plan.logical_f32_element_count)?,
                    budget.memory_budget_bytes
                )));
            }
            let (f32_preview, f32_preview_truncated, total_bytes_read_from_disk) =
                materialize_read_plan_f32_le_for_execution(mmap, plan, Some(max_f32))?;
            let mut preview = QueryExecutionPreview::with_operation_and_io(
                total_bytes_read_from_disk,
                f32_preview,
                f32_preview_truncated,
                OperationPreviewFields::default(),
                Some(MemoryStrategy::CappedInMemory.as_str()),
                None,
                None,
            );
            attach_budget_fields(&mut preview, budget, plan);
            Ok(preview)
        }
        Some(op) => {
            if let Some(kind) = scalar_reduction_kind(op) {
                let folded = fold_read_plan_scalar_operation(mmap, plan, max_f32, kind)?;
                return Ok(fold_outcome_to_preview(
                    folded,
                    MemoryStrategy::StreamingFold,
                    budget,
                    plan,
                ));
            }
            let kind = ReductionKind::from(op);
            let folded = fold_read_plan_partial_operation(mmap, plan, max_f32, kind, op.axes())?;
            Ok(fold_outcome_to_preview(
                folded,
                MemoryStrategy::StreamingFold,
                budget,
                plan,
            ))
        }
    }
}
