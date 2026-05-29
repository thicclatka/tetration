//! Two-pass element-wise transforms (`transform` wire key).
//!
//! Pass 1 ([`stats`]) folds statistics over the logical selection. Pass 2 ([`apply`])
//! decodes chunks again and rewrites each element in place (RAM or spill). Division by
//! zero (or an invalid `sqrt` shift) yields **NaN** and is recorded in
//! [`warnings::TransformWarnings`] for the execution preview.

mod apply;
mod stats;
mod target;
mod warnings;

pub use crate::query::types::TransformMethod;

use std::path::Path;

use crate::query::dispatch;
use crate::query::engine::{
    budget::{ExecutionBudget, MemoryStrategy},
    spill_policy::SpillPathAllowlist,
};
use crate::query::fold::fold_policy::FoldIoPolicy;
use crate::query::materialize::DecodePreviewBundle;
use crate::query::types::{Operation, OperationPreviewFields, ReadPlan, TetError, WriteHints};
use crate::utils::dtype::ElementDtype;

/// Result of [`run_transform`]: decode preview, pass-1 stats, spill metadata, and warnings.
pub(crate) struct TransformOutcome {
    pub total_bytes_read_from_disk: u64,
    pub strategy: MemoryStrategy,
    pub spill_path: Option<String>,
    pub spill_bytes: Option<u64>,
    pub bundle: DecodePreviewBundle,
    pub operation: OperationPreviewFields,
}

/// Inputs shared by pass 1 and pass 2 of a transform operation.
pub(crate) struct TransformRunInput<'a> {
    pub mmap: &'a [u8],
    pub plan: &'a ReadPlan,
    pub op: &'a Operation,
    pub write: Option<&'a WriteHints>,
    pub max_preview: usize,
    pub budget: ExecutionBudget,
    pub dtype: ElementDtype,
    pub spill_allowlist: &'a SpillPathAllowlist,
    pub tet_path: Option<&'a Path>,
    pub fold_policy: FoldIoPolicy,
}

struct Pass2Outcome {
    strategy: MemoryStrategy,
    spill_path: Option<String>,
    spill_bytes: Option<u64>,
    bundle: DecodePreviewBundle,
    pass2_bytes: u64,
    warnings: warnings::TransformWarnings,
}

const ERR_TRANSFORM_FLOAT: &str = "transform requires f32 or f64 datasets";

fn capped_preview<T: Copy>(values: &[T], max_preview: usize, logical_len: usize) -> (Vec<T>, bool) {
    let truncated = logical_len > max_preview;
    let preview = if max_preview == 0 {
        Vec::new()
    } else {
        values[..max_preview.min(logical_len)].to_vec()
    };
    (preview, truncated)
}

fn transform_ram_outcome(
    bundle: DecodePreviewBundle,
    bytes: u64,
    warnings: warnings::TransformWarnings,
) -> Pass2Outcome {
    Pass2Outcome {
        strategy: MemoryStrategy::TransformRam,
        spill_path: None,
        spill_bytes: None,
        bundle,
        pass2_bytes: bytes,
        warnings,
    }
}

fn run_pass2(
    input: &TransformRunInput<'_>,
    stats: &stats::TransformStats,
    output: target::ResolvedTransformOutput,
) -> Result<Pass2Outcome, TetError> {
    let logical_len = input.plan.logical_f32_element_count;
    let mut warnings = warnings::TransformWarnings::default();
    match (input.dtype, output) {
        (ElementDtype::F32, target::ResolvedTransformOutput::Ram) => {
            let (values, bytes) = apply::transform_read_plan_f32_le_ram(
                input.mmap,
                input.plan,
                stats,
                &mut warnings,
            )?;
            let (preview, truncated) = capped_preview(&values, input.max_preview, logical_len);
            Ok(transform_ram_outcome(
                DecodePreviewBundle::f32_preview(preview, truncated),
                bytes,
                warnings,
            ))
        }
        (ElementDtype::F64, target::ResolvedTransformOutput::Ram) => {
            let (values, bytes) = apply::transform_read_plan_f64_le_ram(
                input.mmap,
                input.plan,
                stats,
                &mut warnings,
            )?;
            let (preview, truncated) = capped_preview(&values, input.max_preview, logical_len);
            Ok(transform_ram_outcome(
                DecodePreviewBundle::f64_preview(preview, truncated),
                bytes,
                warnings,
            ))
        }
        (ElementDtype::F32 | ElementDtype::F64, target::ResolvedTransformOutput::Spill(path)) => {
            spill_pass2(input, stats, &path)
        }
        _ => Err(TetError::Validation(ERR_TRANSFORM_FLOAT.into())),
    }
}

fn spill_pass2(
    input: &TransformRunInput<'_>,
    stats: &stats::TransformStats,
    path: &Path,
) -> Result<Pass2Outcome, TetError> {
    let mut warnings = warnings::TransformWarnings::default();
    let pass2_bytes = match input.dtype {
        ElementDtype::F32 => {
            apply::transform_spill_f32_le(input.mmap, input.plan, path, stats, &mut warnings)?
        }
        ElementDtype::F64 => {
            apply::transform_spill_f64_le(input.mmap, input.plan, path, stats, &mut warnings)?
        }
        _ => {
            return Err(TetError::Validation(ERR_TRANSFORM_FLOAT.into()));
        }
    };
    let spill_bytes = input
        .budget
        .logical_element_bytes(input.dtype, input.plan.logical_f32_element_count)?;
    let bundle = dispatch::spill_export_preview(
        path,
        input.plan.logical_f32_element_count,
        input.max_preview,
        input.dtype,
    )?;
    Ok(Pass2Outcome {
        strategy: MemoryStrategy::TransformSpill,
        spill_path: Some(path.display().to_string()),
        spill_bytes: Some(spill_bytes),
        bundle,
        pass2_bytes,
        warnings,
    })
}

/// Run pass-1 stats collection and pass-2 transform apply.
///
/// # Errors
///
/// Propagates fold, budget, spill-path, and materialize failures.
pub(crate) fn run_transform(input: &TransformRunInput<'_>) -> Result<TransformOutcome, TetError> {
    let (output, _) = target::resolve_transform_output(
        input.write,
        &input.budget,
        input.plan,
        input.dtype,
        input.spill_allowlist,
    )?;
    let (stats, mut operation, pass1_bytes) = stats::collect_transform_stats(
        input.mmap,
        input.plan,
        input.op,
        input.dtype,
        &input.fold_policy,
        input.tet_path,
    )?;
    let pass2 = run_pass2(input, &stats, output)?;
    let total_bytes_read_from_disk = pass1_bytes
        .checked_add(pass2.pass2_bytes)
        .ok_or_else(|| TetError::Validation("total bytes read overflow".into()))?;
    operation.element_count = Some(input.plan.logical_f32_element_count);
    if pass2.warnings.div_by_zero_count > 0 {
        operation.transform_div_by_zero_count = Some(pass2.warnings.div_by_zero_count);
        operation.transform_div_by_zero_indices = Some(pass2.warnings.div_by_zero_indices);
    }
    Ok(TransformOutcome {
        total_bytes_read_from_disk,
        strategy: pass2.strategy,
        spill_path: pass2.spill_path,
        spill_bytes: pass2.spill_bytes,
        bundle: pass2.bundle,
        operation,
    })
}
