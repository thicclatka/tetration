//! Phase 10: optional GPU execution routing (scaffold — CPU path until `tetration-gpu` kernels land).
//!
//! Decode and mmap stay on the host; [`DeviceRoute`] records what was requested vs what ran.

use crate::query::types::{ExecutionDeviceHint, ExecutionHints, Operation, ReadPlan};
use crate::utils::dtype::ElementDtype;

/// Minimum logical selection size for `device: auto` to consider GPU (when enabled).
pub const GPU_AUTO_MIN_LOGICAL_BYTES: u64 = 64 * 1024 * 1024;

/// Resolved device path for one execution (written to `execution` preview JSON).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceRoute {
    pub requested: Option<ExecutionDeviceHint>,
    pub used: &'static str,
    pub fallback_reason: Option<&'static str>,
    pub gpu_reduce: bool,
}

impl DeviceRoute {
    #[must_use]
    pub const fn cpu_only() -> Self {
        Self {
            requested: None,
            used: "cpu",
            fallback_reason: None,
            gpu_reduce: false,
        }
    }
}

/// Pick host vs (future) device reduction for this execute.
#[must_use]
pub fn resolve_device_route(
    hints: Option<&ExecutionHints>,
    plan: &ReadPlan,
    dtype: ElementDtype,
    operation: Option<&Operation>,
) -> DeviceRoute {
    let Some(requested) = hints.and_then(|h| h.device) else {
        return DeviceRoute::cpu_only();
    };

    if operation.is_none() {
        return DeviceRoute {
            requested: Some(requested),
            used: "cpu",
            fallback_reason: Some("preview_or_spill_only"),
            gpu_reduce: false,
        };
    }

    if let Some(op) = operation
        && op.requires_materialize()
    {
        return DeviceRoute {
            requested: Some(requested),
            used: "cpu",
            fallback_reason: Some("tier_c_materialize"),
            gpu_reduce: false,
        };
    }

    if matches!(requested, ExecutionDeviceHint::Cpu) {
        return DeviceRoute {
            requested: Some(requested),
            used: "cpu",
            fallback_reason: None,
            gpu_reduce: false,
        };
    }

    let logical_bytes = u64::try_from(plan.logical_f32_element_count)
        .ok()
        .and_then(|n| dtype.bytes_from_elem_count(n));

    if matches!(requested, ExecutionDeviceHint::Auto) {
        let Some(bytes) = logical_bytes else {
            return DeviceRoute {
                requested: Some(requested),
                used: "cpu",
                fallback_reason: Some("auto_unknown_logical_bytes"),
                gpu_reduce: false,
            };
        };
        if bytes < GPU_AUTO_MIN_LOGICAL_BYTES {
            return DeviceRoute {
                requested: Some(requested),
                used: "cpu",
                fallback_reason: Some("auto_below_size_threshold"),
                gpu_reduce: false,
            };
        }
    }

    if !gpu_backend_available() {
        return DeviceRoute {
            requested: Some(requested),
            used: "cpu",
            fallback_reason: Some("gpu_feature_disabled"),
            gpu_reduce: false,
        };
    }

    // Feature enabled but kernels not wired yet.
    DeviceRoute {
        requested: Some(requested),
        used: "cpu",
        fallback_reason: Some("gpu_not_implemented"),
        gpu_reduce: false,
    }
}

#[must_use]
pub fn gpu_backend_available() -> bool {
    cfg!(feature = "tetration-gpu")
}

pub(crate) fn attach_device_fields(
    preview: &mut crate::query::types::QueryExecutionPreview,
    route: DeviceRoute,
) {
    preview.device_requested = route.requested.map(ExecutionDeviceHint::to_token);
    preview.device_used = Some(route.used);
    preview.device_fallback_reason = route.fallback_reason.map(str::to_string);
    preview.device_gpu_reduce = Some(route.gpu_reduce);
}
