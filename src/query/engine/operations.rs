#![allow(clippy::too_many_arguments)]

use std::path::Path;

use crate::query::types::{
    Operation, OperationPreviewFields, OutputHint, OutputHints, QueryExecutionPreview, ReadPlan,
    TetError,
};
use crate::utils::dtype::ElementDtype;

use super::budget::{ExecutionBudget, MemoryStrategy};
use super::dispatch::{
    materialize_for_execution, partial_fold, scalar_fold, spill_export_preview,
    spill_full_selection,
};
use super::materialize::{
    DecodePreviewBundle, materialize_logical_selection, preview_from_materialized,
};
use super::materialize_stats::run_tier_c_operation;
use super::reduction::ReductionKind;
use super::spill_policy::SpillPathAllowlist;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OperationMaterializeTier {
    Streaming,
    MaterializeRequired,
}

impl Operation {
    fn materialize_tier(&self) -> OperationMaterializeTier {
        match self {
            Self::Median { .. } | Self::Quantile { .. } | Self::Histogram { .. } => {
                OperationMaterializeTier::MaterializeRequired
            }
            _ => OperationMaterializeTier::Streaming,
        }
    }
}

fn materialized_io(
    materialized: &super::materialize::MaterializedLogical,
) -> (u64, MemoryStrategy) {
    match materialized {
        super::materialize::MaterializedLogical::F32 {
            total_bytes_read_from_disk,
            strategy,
            ..
        }
        | super::materialize::MaterializedLogical::F64 {
            total_bytes_read_from_disk,
            strategy,
            ..
        }
        | super::materialize::MaterializedLogical::I32 {
            total_bytes_read_from_disk,
            strategy,
            ..
        }
        | super::materialize::MaterializedLogical::I64 {
            total_bytes_read_from_disk,
            strategy,
            ..
        } => (*total_bytes_read_from_disk, *strategy),
    }
}

fn preview_from_bundle(
    total_bytes_read_from_disk: u64,
    bundle: DecodePreviewBundle,
    operation: OperationPreviewFields,
    memory_strategy: Option<&'static str>,
    spill_f32_path: Option<String>,
    spill_f32_bytes: Option<u64>,
) -> QueryExecutionPreview {
    QueryExecutionPreview::with_operation_and_io(
        total_bytes_read_from_disk,
        bundle.f32,
        bundle.f32_truncated,
        bundle.f64,
        bundle.f64_truncated,
        bundle.i32,
        bundle.i32_truncated,
        bundle.i64,
        bundle.i64_truncated,
        operation,
        memory_strategy,
        spill_f32_path,
        spill_f32_bytes,
    )
}

fn attach_budget_fields(
    preview: &mut QueryExecutionPreview,
    budget: ExecutionBudget,
    plan: &ReadPlan,
    dtype: ElementDtype,
) {
    preview.memory_budget_bytes = Some(budget.memory_budget_bytes);
    preview.host_available_ram_bytes = budget.host_available_ram_bytes;
    preview.logical_selection_bytes = budget
        .logical_element_bytes(dtype, plan.logical_f32_element_count)
        .ok();
    if dtype == ElementDtype::F32 {
        preview.logical_selection_f32_bytes = preview.logical_selection_bytes;
    }
}

fn fold_outcome_to_preview(
    folded: super::fold::FoldPlanOutcome,
    strategy: MemoryStrategy,
    budget: ExecutionBudget,
    plan: &ReadPlan,
    dtype: ElementDtype,
) -> QueryExecutionPreview {
    let mut preview = preview_from_bundle(
        folded.total_bytes_read_from_disk,
        DecodePreviewBundle {
            f32: folded.f32_preview,
            f64: folded.f64_preview,
            i32: folded.i32_preview,
            i64: folded.i64_preview,
            f32_truncated: folded.f32_preview_truncated,
            f64_truncated: folded.f64_preview_truncated,
            i32_truncated: folded.i32_preview_truncated,
            i64_truncated: folded.i64_preview_truncated,
        },
        folded.operation,
        Some(strategy.as_str()),
        None,
        None,
    );
    attach_budget_fields(&mut preview, budget, plan, dtype);
    preview
}

fn run_materialize_required_operation(
    mmap: &[u8],
    plan: &ReadPlan,
    op: &Operation,
    max_preview: usize,
    budget: &ExecutionBudget,
    allowlist: &SpillPathAllowlist,
    dtype: ElementDtype,
) -> Result<QueryExecutionPreview, TetError> {
    let materialized = materialize_logical_selection(mmap, plan, budget, allowlist, dtype)?;
    let bundle =
        preview_from_materialized(&materialized, plan.logical_f32_element_count, max_preview)?;
    let operation = run_tier_c_operation(&materialized, plan, op)?;
    let (total_bytes_read_from_disk, strategy) = materialized_io(&materialized);
    let mut preview = preview_from_bundle(
        total_bytes_read_from_disk,
        bundle,
        operation,
        Some(strategy.as_str()),
        None,
        None,
    );
    attach_budget_fields(&mut preview, *budget, plan, dtype);
    Ok(preview)
}

fn scalar_reduction_kind(op: &Operation) -> Option<ReductionKind> {
    op.axes().is_empty().then(|| ReductionKind::from(op))
}

fn spill_requested(output: Option<&OutputHints>) -> Option<&str> {
    match output.and_then(|o| o.preferred.as_ref()) {
        Some(OutputHint::SpillArray { handle }) => Some(handle.as_str()),
        _ => None,
    }
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
    let elem_dtype = ElementDtype::from_wire(dtype)?;

    match operation {
        None => build_decode_preview(
            mmap,
            plan,
            output,
            max_f32,
            budget,
            spill_allowlist,
            elem_dtype,
        ),
        Some(op) => {
            build_operation_preview(mmap, plan, op, max_f32, budget, spill_allowlist, elem_dtype)
        }
    }
}

fn build_decode_preview(
    mmap: &[u8],
    plan: &ReadPlan,
    output: Option<&OutputHints>,
    max_preview: usize,
    budget: ExecutionBudget,
    spill_allowlist: Option<&SpillPathAllowlist>,
    dtype: ElementDtype,
) -> Result<QueryExecutionPreview, TetError> {
    if let Some(spill_path) = spill_requested(output) {
        let path = Path::new(spill_path);
        let policy = spill_allowlist.ok_or_else(|| {
            TetError::Validation(
                "spill output requires a spill path allowlist (pass `--tet` so defaults apply)"
                    .into(),
            )
        })?;
        let resolved = policy.validate(path)?;
        let total_bytes_read_from_disk = spill_full_selection(mmap, plan, &resolved, dtype)?;
        let bundle = spill_export_preview(
            &resolved,
            plan.logical_f32_element_count,
            max_preview,
            dtype,
        )?;
        let spill_bytes = budget.logical_element_bytes(dtype, plan.logical_f32_element_count)?;
        let mut preview = preview_from_bundle(
            total_bytes_read_from_disk,
            bundle,
            OperationPreviewFields::default(),
            Some(MemoryStrategy::MmapSpill.as_str()),
            Some(resolved.display().to_string()),
            Some(spill_bytes),
        );
        attach_budget_fields(&mut preview, budget, plan, dtype);
        return Ok(preview);
    }
    if budget.full_tensor_exceeds_budget(plan, dtype)? && max_preview == 0 {
        return Err(TetError::Validation(format!(
            "logical selection ({} elements, {} bytes) exceeds memory_budget_bytes ({}); \
             use a positive preview cap, an `operation`, output spill, or raise execution.memory_budget_bytes / memory_budget_percent_bps",
            plan.logical_f32_element_count,
            budget.logical_element_bytes(dtype, plan.logical_f32_element_count)?,
            budget.memory_budget_bytes
        )));
    }
    let (bundle, total_bytes) = materialize_for_execution(mmap, plan, Some(max_preview), dtype)?;
    let mut preview = preview_from_bundle(
        total_bytes,
        bundle,
        OperationPreviewFields::default(),
        Some(MemoryStrategy::CappedInMemory.as_str()),
        None,
        None,
    );
    attach_budget_fields(&mut preview, budget, plan, dtype);
    Ok(preview)
}

fn build_operation_preview(
    mmap: &[u8],
    plan: &ReadPlan,
    op: &Operation,
    max_preview: usize,
    budget: ExecutionBudget,
    spill_allowlist: Option<&SpillPathAllowlist>,
    dtype: ElementDtype,
) -> Result<QueryExecutionPreview, TetError> {
    if op.materialize_tier() == OperationMaterializeTier::MaterializeRequired {
        let policy = spill_allowlist.ok_or_else(|| {
            TetError::Validation(
                "materialize-required operation needs a spill path allowlist (pass `--tet` so defaults apply)"
                    .into(),
            )
        })?;
        return run_materialize_required_operation(
            mmap,
            plan,
            op,
            max_preview,
            &budget,
            policy,
            dtype,
        );
    }
    if let Some(kind) = scalar_reduction_kind(op) {
        let folded = scalar_fold(mmap, plan, max_preview, kind, dtype)?;
        return Ok(fold_outcome_to_preview(
            folded,
            MemoryStrategy::StreamingFold,
            budget,
            plan,
            dtype,
        ));
    }
    let kind = ReductionKind::from(op);
    let folded = partial_fold(mmap, plan, max_preview, kind, op.axes(), dtype)?;
    Ok(fold_outcome_to_preview(
        folded,
        MemoryStrategy::StreamingFold,
        budget,
        plan,
        dtype,
    ))
}
