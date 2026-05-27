//! Phase 10: optional device scalar `f32` reductions after host decode.
//!
//! - **10b** `tetration-gpu` — CUDA (`cuda` / `auto` on Linux+Windows with NVIDIA)
//! - **10c** `tetration-metal` — Metal (`metal` / `auto` on macOS)

mod scalar_fold;

#[cfg(feature = "tetration-gpu")]
mod cuda;

#[cfg(all(feature = "tetration-metal", target_os = "macos"))]
mod metal;

use crate::query::device::DeviceRoute;
use crate::query::fold::FoldPlanOutcome;
use crate::query::fold::reduction::ReductionKind;
use crate::query::types::ReadPlan;

/// Host decode + device reduce when [`DeviceRoute::gpu_reduce`] is set.
pub(crate) fn try_scalar_f32_fold(
    mmap: &[u8],
    plan: &ReadPlan,
    max_preview: usize,
    kind: ReductionKind,
    route: DeviceRoute,
) -> Result<(FoldPlanOutcome, DeviceRoute), &'static str> {
    if !route.gpu_reduce {
        return Err("gpu_not_requested");
    }
    if !crate::query::device::gpu_supports_scalar_kind(kind) {
        return Err("gpu_unsupported_op");
    }

    match route.used {
        #[cfg(all(feature = "tetration-metal", target_os = "macos"))]
        "metal" => {
            return scalar_fold::try_scalar_f32_fold_metal(mmap, plan, max_preview, kind, route);
        }
        #[cfg(feature = "tetration-gpu")]
        "cuda" => {
            return scalar_fold::try_scalar_f32_fold_cuda(mmap, plan, max_preview, kind, route);
        }
        _ => {}
    }

    let _ = (mmap, plan, max_preview, kind, route);
    Err("gpu_feature_disabled")
}

/// Fraction of reported device memory allowed for one H2D buffer (rest for driver + scratch).
#[cfg(any(feature = "tetration-gpu", feature = "tetration-metal"))]
pub(crate) const GPU_VRAM_BUFFER_FRACTION: f64 = 0.85;
