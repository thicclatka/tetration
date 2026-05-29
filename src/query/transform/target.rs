//! Resolve transform `write` hints to RAM or a spill path.

use std::path::{Path, PathBuf};

use crate::query::engine::{budget::ExecutionBudget, spill_policy::SpillPathAllowlist};
use crate::query::types::{ReadPlan, TetError, WriteHints, WriteTarget};
use crate::utils::dtype::ElementDtype;

pub(crate) enum ResolvedTransformOutput {
    Ram,
    Spill(PathBuf),
}

pub(crate) fn resolve_transform_output(
    write: Option<&WriteHints>,
    budget: &ExecutionBudget,
    plan: &ReadPlan,
    dtype: ElementDtype,
    allowlist: &SpillPathAllowlist,
) -> Result<(ResolvedTransformOutput, WriteTarget), TetError> {
    let target = write.map_or(WriteTarget::Switch, |w| w.target);
    let path_hint = write.and_then(|w| w.path.as_deref());

    if target == WriteTarget::Sidecar {
        return Err(TetError::Validation(
            "sidecar transform output is not implemented yet; use `write`: `spill` or `switch`"
                .into(),
        ));
    }

    let exceeds = budget.full_tensor_exceeds_budget(plan, dtype)?;

    match target {
        WriteTarget::Ram => {
            if exceeds {
                return Err(TetError::Validation(format!(
                    "logical selection ({} elements, {} bytes) exceeds memory_budget_bytes ({}); \
                     use `write`: `switch` or `spill`, or raise execution.memory_budget_bytes",
                    plan.logical_f32_element_count,
                    budget.logical_element_bytes(dtype, plan.logical_f32_element_count)?,
                    budget.memory_budget_bytes
                )));
            }
            Ok((ResolvedTransformOutput::Ram, WriteTarget::Ram))
        }
        WriteTarget::Spill => {
            let path = resolve_spill_path(path_hint, allowlist)?;
            Ok((ResolvedTransformOutput::Spill(path), WriteTarget::Spill))
        }
        WriteTarget::Switch => {
            if exceeds {
                let path = resolve_spill_path(path_hint, allowlist)?;
                Ok((ResolvedTransformOutput::Spill(path), WriteTarget::Switch))
            } else {
                Ok((ResolvedTransformOutput::Ram, WriteTarget::Switch))
            }
        }
        WriteTarget::Sidecar => unreachable!("handled above"),
    }
}

fn resolve_spill_path(
    path_hint: Option<&str>,
    allowlist: &SpillPathAllowlist,
) -> Result<PathBuf, TetError> {
    match path_hint {
        Some(handle) => allowlist.validate(Path::new(handle)),
        None => allowlist.allocate_temp_spill_path(),
    }
}
