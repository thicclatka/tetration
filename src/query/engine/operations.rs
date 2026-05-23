use std::cmp::Ordering;
use std::path::Path;

use memmap2::MmapMut;

use crate::catalog::DTYPE_F32;
use crate::query::types::{
    Operation, OperationPreviewFields, OutputHint, OutputHints, QueryExecutionPreview, ReadPlan,
    TetError,
};

use super::budget::{ExecutionBudget, MemoryStrategy};
use super::materialize::{
    FoldPlanOutcome, LogicalF32Backing, fold_read_plan_scalar_operation,
    materialize_logical_selection, materialize_read_plan_f32_le, preview_from_materialized,
    preview_from_spill_export_file, spill_read_plan_f32_le,
};
use super::parallel::materialize_read_plan_f32_le_parallel;
use super::partial_fold::fold_read_plan_partial_operation;
use super::reduction::ReductionKind;
use super::spill_policy::SpillPathAllowlist;

// --- Operation execution tier (streaming fold vs full materialize) ---

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OperationMaterializeTier {
    Streaming,
    MaterializeRequired,
}

impl Operation {
    fn materialize_tier(&self) -> OperationMaterializeTier {
        match self {
            Self::Median { .. } => OperationMaterializeTier::MaterializeRequired,
            _ => OperationMaterializeTier::Streaming,
        }
    }
}

// --- Tier-C stats over materialized logical selections ---

fn median_f32(values: &mut [f32]) -> Result<f64, TetError> {
    if values.is_empty() {
        return Err(TetError::Validation(
            "median requires at least one element".into(),
        ));
    }
    let cmp = |a: &f32, b: &f32| a.partial_cmp(b).unwrap_or(Ordering::Equal);
    let n = values.len();
    let mid = n / 2;
    if n.is_multiple_of(2) {
        values.select_nth_unstable_by(mid, cmp);
        let hi = values[mid];
        values.select_nth_unstable_by(mid - 1, cmp);
        Ok(f64::from(f32::midpoint(values[mid - 1], hi)))
    } else {
        values.select_nth_unstable_by(mid, cmp);
        Ok(f64::from(values[mid]))
    }
}

fn median_f32_spill_file(path: &Path) -> Result<f64, TetError> {
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .map_err(|e| TetError::Validation(format!("temp spill open failed: {e}")))?;
    let mut mmap = unsafe {
        MmapMut::map_mut(&file)
            .map_err(|e| TetError::Validation(format!("temp spill mmap mut failed: {e}")))?
    };
    let slice = bytemuck::cast_slice_mut(mmap.as_mut());
    median_f32(slice)
}

fn run_materialize_required_operation(
    mmap: &[u8],
    plan: &ReadPlan,
    op: &Operation,
    max_f32: usize,
    budget: &ExecutionBudget,
    allowlist: &SpillPathAllowlist,
) -> Result<QueryExecutionPreview, TetError> {
    if !op.axes().is_empty() {
        return Err(TetError::Validation(format!(
            "{op:?} with non-empty axes is not supported yet (scalar `axes: []` only)"
        )));
    }
    let mut materialized = materialize_logical_selection(mmap, plan, budget, allowlist)?;
    let element_count = plan.logical_f32_element_count;
    let (f32_preview, f32_preview_truncated) =
        preview_from_materialized(&materialized.backing, element_count, max_f32)?;

    let operation = match op {
        Operation::Median { .. } => {
            let median = match &mut materialized.backing {
                LogicalF32Backing::InMemory(v) => median_f32(v)?,
                LogicalF32Backing::TempSpill(temp) => median_f32_spill_file(temp.path())?,
            };
            OperationPreviewFields {
                element_count: Some(element_count),
                median: Some(median),
                ..OperationPreviewFields::default()
            }
        }
        _ => {
            return Err(TetError::Validation(format!(
                "unsupported materialize-required operation: {op:?}"
            )));
        }
    };

    let mut preview = QueryExecutionPreview::with_operation_and_io(
        materialized.total_bytes_read_from_disk,
        f32_preview,
        f32_preview_truncated,
        operation,
        Some(materialized.strategy.as_str()),
        None,
        None,
    );
    preview.memory_budget_bytes = Some(budget.memory_budget_bytes);
    preview.host_available_ram_bytes = budget.host_available_ram_bytes;
    preview.logical_selection_f32_bytes = budget
        .logical_f32_bytes(plan.logical_f32_element_count)
        .ok();
    Ok(preview)
}

// --- Execution preview routing ---

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

pub(super) struct ExecutionPreviewInput<'a> {
    pub mmap: &'a [u8],
    pub plan: &'a ReadPlan,
    pub dtype: u32,
    pub operation: Option<&'a Operation>,
    pub output: Option<&'a OutputHints>,
    pub max_f32: usize,
    pub budget: ExecutionBudget,
    pub spill_allowlist: Option<&'a SpillPathAllowlist>,
}

pub(super) fn build_execution_preview(
    input: &ExecutionPreviewInput<'_>,
) -> Result<QueryExecutionPreview, TetError> {
    let ExecutionPreviewInput {
        mmap,
        plan,
        dtype,
        operation,
        output,
        max_f32,
        budget,
        spill_allowlist,
    } = *input;
    if dtype != DTYPE_F32 {
        return Err(TetError::Validation(
            "f32 preview requires dataset dtype f32 (DTYPE_F32 = 1)".into(),
        ));
    }

    match operation {
        None => {
            if let Some(spill_path) = spill_requested(output) {
                let path = std::path::Path::new(spill_path);
                let policy = spill_allowlist.ok_or_else(|| {
                    TetError::Validation(
                        "spill output requires a spill path allowlist (pass `--tet` so defaults apply)"
                            .into(),
                    )
                })?;
                let resolved = policy.validate(path)?;
                let total_bytes_read_from_disk = spill_read_plan_f32_le(mmap, plan, &resolved)?;
                let (f32_preview, f32_preview_truncated) = preview_from_spill_export_file(
                    &resolved,
                    plan.logical_f32_element_count,
                    max_f32,
                )?;
                let spill_bytes = budget.logical_f32_bytes(plan.logical_f32_element_count)?;
                let mut preview = QueryExecutionPreview::with_operation_and_io(
                    total_bytes_read_from_disk,
                    f32_preview,
                    f32_preview_truncated,
                    OperationPreviewFields::default(),
                    Some(MemoryStrategy::MmapSpill.as_str()),
                    Some(resolved.display().to_string()),
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
            if op.materialize_tier() == OperationMaterializeTier::MaterializeRequired {
                let policy = spill_allowlist.ok_or_else(|| {
                    TetError::Validation(
                        "materialize-required operation needs a spill path allowlist (pass `--tet` so defaults apply)"
                            .into(),
                    )
                })?;
                return run_materialize_required_operation(
                    mmap, plan, op, max_f32, &budget, policy,
                );
            }
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
