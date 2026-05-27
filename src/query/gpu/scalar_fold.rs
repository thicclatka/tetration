//! Host materialize + shared helpers for device scalar `f32` folds.

#[cfg(any(feature = "tetration-gpu", feature = "tetration-metal"))]
use crate::query::device::DeviceRoute;
#[cfg(any(feature = "tetration-gpu", feature = "tetration-metal"))]
use crate::query::dispatch;
#[cfg(any(feature = "tetration-gpu", feature = "tetration-metal"))]
use crate::query::fold::reduction::ReductionKind;
#[cfg(any(feature = "tetration-gpu", feature = "tetration-metal"))]
use crate::query::fold::{self, FoldPlanOutcome};
#[cfg(any(feature = "tetration-gpu", feature = "tetration-metal"))]
use crate::query::materialize::{self, parallel};
#[cfg(any(feature = "tetration-gpu", feature = "tetration-metal"))]
use crate::query::types::{ReadPlan, TetError};

#[cfg(feature = "tetration-gpu")]
use super::cuda;

#[cfg(all(feature = "tetration-metal", target_os = "macos"))]
use super::metal;

#[cfg(feature = "tetration-gpu")]
pub(crate) fn try_scalar_f32_fold_cuda(
    mmap: &[u8],
    plan: &ReadPlan,
    max_preview: usize,
    kind: ReductionKind,
    route: DeviceRoute,
) -> Result<(FoldPlanOutcome, DeviceRoute), &'static str> {
    let device_index = route.cuda_device.unwrap_or(0);
    let n = plan.logical_f32_element_count;
    let logical_bytes = n
        .checked_mul(4)
        .ok_or("gpu_logical_bytes_overflow")?;
    if let Err(reason) = cuda::vram_check(device_index, logical_bytes) {
        return Err(reason);
    }

    let mut host = vec![0.0_f32; n];
    let total_bytes_read_from_disk =
        materialize_host_f32(mmap, plan, &mut host).map_err(|_| "gpu_host_decode_failed")?;

    let scalar =
        cuda::reduce_f32_scalar(&host, kind, device_index).map_err(|_| "gpu_runtime_error")?;

    build_gpu_fold_outcome(
        route,
        "cuda",
        Some(device_index),
        host,
        max_preview,
        n,
        total_bytes_read_from_disk,
        scalar.into(),
    )
}

#[cfg(all(feature = "tetration-metal", target_os = "macos"))]
pub(crate) fn try_scalar_f32_fold_metal(
    mmap: &[u8],
    plan: &ReadPlan,
    max_preview: usize,
    kind: ReductionKind,
    route: DeviceRoute,
) -> Result<(FoldPlanOutcome, DeviceRoute), &'static str> {
    let n = plan.logical_f32_element_count;
    let logical_bytes = n
        .checked_mul(4)
        .ok_or("gpu_logical_bytes_overflow")?;
    if let Err(reason) = metal::vram_check(logical_bytes) {
        return Err(reason);
    }

    let mut host = vec![0.0_f32; n];
    let total_bytes_read_from_disk =
        materialize_host_f32(mmap, plan, &mut host).map_err(|_| "gpu_host_decode_failed")?;

    let scalar = metal::reduce_f32_scalar(&host, kind).map_err(|_| "gpu_runtime_error")?;

    build_gpu_fold_outcome(
        route,
        "metal",
        None,
        host,
        max_preview,
        n,
        total_bytes_read_from_disk,
        scalar.into(),
    )
}

#[cfg(any(feature = "tetration-gpu", feature = "tetration-metal"))]
fn build_gpu_fold_outcome(
    route: DeviceRoute,
    used: &'static str,
    cuda_device: Option<usize>,
    host: Vec<f32>,
    max_preview: usize,
    n: usize,
    total_bytes_read_from_disk: u64,
    operation: crate::query::types::OperationPreviewFields,
) -> Result<(FoldPlanOutcome, DeviceRoute), &'static str> {
    let preview_cap = max_preview.min(n);
    let preview = if max_preview == 0 {
        Vec::new()
    } else {
        host[..preview_cap].to_vec()
    };
    fold::shared::validate_fold_preview(n > 0, &preview, preview_cap)
        .map_err(|_| "gpu_empty_selection")?;

    let outcome = fold::shared::build_fold_plan_outcome(
        preview,
        max_preview,
        n,
        total_bytes_read_from_disk,
        operation,
    );
    let ok_route = DeviceRoute {
        requested: route.requested,
        used,
        fallback_reason: None,
        gpu_reduce: true,
        cuda_device,
    };
    Ok((outcome, ok_route))
}

#[cfg(any(feature = "tetration-gpu", feature = "tetration-metal"))]
fn materialize_host_f32(
    mmap: &[u8],
    plan: &ReadPlan,
    dst: &mut [f32],
) -> Result<u64, TetError> {
    if plan.chunks.len() > 1 {
        parallel::materialize_read_plan_f32_le_into_parallel(mmap, plan, None, dst)?;
    } else {
        materialize::materialize_read_plan_f32_le_into(mmap, plan, None, dst)?;
    }
    let bytes = plan.chunks.iter().map(|c| c.stored_byte_len);
    dispatch::sum_chunk_read_bytes(bytes)
}
