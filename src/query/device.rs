//! Phase 10: optional GPU execution routing (CUDA / Metal scalar `f32` reductions).
//!
//! Decode and mmap stay on the host; [`DeviceRoute`] records what was requested vs what ran.

use crate::query::fold::reduction::ReductionKind;
use crate::query::types::{ExecutionDeviceHint, ExecutionHints, Operation, ReadPlan};
use crate::utils::dtype::ElementDtype;
use crate::utils::host_memory;

/// Minimum logical selection size for `device: auto` to consider GPU (when enabled).
pub const GPU_AUTO_MIN_LOGICAL_BYTES: u64 = 64 * 1024 * 1024;

/// Max share of reported host RAM for the dense host `f32` materialize buffer before GPU reduce.
pub const GPU_HOST_MATERIALIZE_RAM_FRACTION: f64 = 0.85;

/// When host RAM cannot be probed, refuse GPU materialize above this logical size.
pub const GPU_HOST_MATERIALIZE_UNKNOWN_HOST_CAP_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// True when `logical_bytes` exceeds the host materialize budget (GPU path must decode to a dense vec).
#[must_use]
pub fn host_materialize_exceeds(logical_bytes: u64, host_available: Option<u64>) -> bool {
    let Some(available) = host_available else {
        return logical_bytes > GPU_HOST_MATERIALIZE_UNKNOWN_HOST_CAP_BYTES;
    };
    let limit = ((available as f64) * GPU_HOST_MATERIALIZE_RAM_FRACTION) as u64;
    logical_bytes > limit
}

/// Whether the GPU scalar path may allocate a dense host buffer for this selection.
#[must_use]
pub(crate) fn host_materialize_fits(logical_bytes: u64) -> bool {
    !host_materialize_exceeds(logical_bytes, host_memory::available_memory_bytes())
}

/// Resolved device path for one execution (written to `execution` preview JSON).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceRoute {
    pub requested: Option<ExecutionDeviceHint>,
    pub used: &'static str,
    pub fallback_reason: Option<&'static str>,
    pub gpu_reduce: bool,
    /// CUDA device index when [`Self::used`] is `"cuda"`.
    pub cuda_device: Option<usize>,
}

impl DeviceRoute {
    #[must_use]
    pub const fn cpu_only() -> Self {
        Self {
            requested: None,
            used: "cpu",
            fallback_reason: None,
            gpu_reduce: false,
            cuda_device: None,
        }
    }

    #[must_use]
    pub const fn cpu_fallback(
        requested: Option<ExecutionDeviceHint>,
        reason: &'static str,
    ) -> Self {
        Self {
            requested,
            used: "cpu",
            fallback_reason: Some(reason),
            gpu_reduce: false,
            cuda_device: None,
        }
    }
}

#[must_use]
pub fn cuda_backend_available() -> bool {
    cfg!(feature = "tetration-gpu")
}

/// Apple Metal path (`tetration-metal`, macOS only).
#[must_use]
pub fn metal_backend_available() -> bool {
    cfg!(all(feature = "tetration-metal", target_os = "macos"))
}

#[must_use]
pub fn gpu_backend_available() -> bool {
    cuda_backend_available() || metal_backend_available()
}

/// Pick host vs device reduction for this execute.
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
        return DeviceRoute::cpu_fallback(Some(requested), "preview_or_spill_only");
    }

    if let Some(op) = operation
        && op.requires_materialize()
    {
        return DeviceRoute::cpu_fallback(Some(requested), "tier_c_materialize");
    }

    if matches!(requested, ExecutionDeviceHint::Cpu) {
        return DeviceRoute {
            requested: Some(requested),
            used: "cpu",
            fallback_reason: None,
            gpu_reduce: false,
            cuda_device: None,
        };
    }

    if dtype != ElementDtype::F32 {
        return DeviceRoute::cpu_fallback(Some(requested), "gpu_unsupported_dtype");
    }

    let scalar_kind = operation.and_then(|op| {
        if op.axes().is_empty() && !op.requires_materialize() {
            Some(ReductionKind::from(op))
        } else {
            None
        }
    });
    if scalar_kind.is_none_or(|k| !gpu_supports_scalar_kind(k)) {
        return DeviceRoute::cpu_fallback(Some(requested), "gpu_unsupported_op");
    }

    let logical_bytes = u64::try_from(plan.logical_f32_element_count)
        .ok()
        .and_then(|n| dtype.bytes_from_elem_count(n));

    if matches!(requested, ExecutionDeviceHint::Auto) {
        let Some(bytes) = logical_bytes else {
            return DeviceRoute::cpu_fallback(Some(requested), "auto_unknown_logical_bytes");
        };
        if bytes < GPU_AUTO_MIN_LOGICAL_BYTES {
            return DeviceRoute::cpu_fallback(Some(requested), "auto_below_size_threshold");
        }
    }

    match requested {
        ExecutionDeviceHint::Metal => {
            if !metal_backend_available() {
                return DeviceRoute::cpu_fallback(Some(requested), "gpu_feature_disabled");
            }
            gpu_scalar_route(Some(requested), "metal", None, logical_bytes)
        }
        ExecutionDeviceHint::Cuda(idx) => {
            if !cuda_backend_available() {
                return DeviceRoute::cpu_fallback(Some(requested), "gpu_feature_disabled");
            }
            gpu_scalar_route(Some(requested), "cuda", Some(idx), logical_bytes)
        }
        ExecutionDeviceHint::Auto => auto_device_route(Some(requested), logical_bytes),
        ExecutionDeviceHint::Cpu => unreachable!("handled above"),
    }
}

/// `device: auto` — Metal on macOS when enabled, else CUDA when enabled.
fn auto_device_route(
    requested: Option<ExecutionDeviceHint>,
    logical_bytes: Option<u64>,
) -> DeviceRoute {
    if metal_backend_available() {
        return gpu_scalar_route(requested, "metal", None, logical_bytes);
    }
    if cuda_backend_available() {
        return gpu_scalar_route(requested, "cuda", Some(0), logical_bytes);
    }
    DeviceRoute::cpu_fallback(requested, "gpu_feature_disabled")
}

fn gpu_scalar_route(
    requested: Option<ExecutionDeviceHint>,
    used: &'static str,
    cuda_device: Option<usize>,
    logical_bytes: Option<u64>,
) -> DeviceRoute {
    if let Some(bytes) = logical_bytes
        && !host_materialize_fits(bytes)
    {
        return DeviceRoute::cpu_fallback(requested, "gpu_host_materialize_exceeded");
    }
    DeviceRoute {
        requested,
        used,
        fallback_reason: None,
        gpu_reduce: true,
        cuda_device,
    }
}

/// Tier-A/B scalar ops implemented on GPU for dense `f32` (population var/std).
#[must_use]
pub(crate) fn gpu_supports_scalar_kind(kind: ReductionKind) -> bool {
    matches!(
        kind,
        ReductionKind::Sum
            | ReductionKind::Mean
            | ReductionKind::Min
            | ReductionKind::Max
            | ReductionKind::Count
            | ReductionKind::Var
            | ReductionKind::Std
    )
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
