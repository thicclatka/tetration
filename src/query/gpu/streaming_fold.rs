//! Per-chunk host decode + device reduce (no full-selection `vec![f32; n]`).
//!
//! When `chunk_count > 1`, decode runs on a worker thread while the main thread
//! reduces the previous chunk on GPU (overlap / async pipeline).

#[cfg(any(
    feature = "tetration-gpu",
    feature = "tetration-metal",
    feature = "tetration-rocm"
))]
use std::sync::mpsc;
#[cfg(any(
    feature = "tetration-gpu",
    feature = "tetration-metal",
    feature = "tetration-rocm"
))]
use std::thread;

#[cfg(any(
    feature = "tetration-gpu",
    feature = "tetration-metal",
    feature = "tetration-rocm"
))]
use crate::query::decode::chunk_decode::{visit_planned_chunk, visit_planned_chunk_f16};
#[cfg(any(
    feature = "tetration-gpu",
    feature = "tetration-metal",
    feature = "tetration-rocm"
))]
use crate::query::device::DeviceRoute;
#[cfg(any(
    feature = "tetration-gpu",
    feature = "tetration-metal",
    feature = "tetration-rocm"
))]
use crate::query::fold::reduction::{
    ArgIndexAccum, ReductionKind, ScalarReductionResult, ValueAccum,
};
#[cfg(any(
    feature = "tetration-gpu",
    feature = "tetration-metal",
    feature = "tetration-rocm"
))]
use crate::query::fold::{self, FoldPlanOutcome};
#[cfg(any(
    feature = "tetration-gpu",
    feature = "tetration-metal",
    feature = "tetration-rocm"
))]
use crate::query::types::{PlannedChunkIo, ReadPlan, TetError};

#[cfg(feature = "tetration-gpu")]
use super::cuda;

#[cfg(feature = "tetration-rocm")]
use super::rocm;

#[cfg(all(feature = "tetration-metal", target_os = "macos"))]
use super::metal;

#[cfg(any(
    feature = "tetration-gpu",
    feature = "tetration-metal",
    feature = "tetration-rocm"
))]
pub(crate) enum StreamingGpuBackend {
    #[cfg(feature = "tetration-gpu")]
    Cuda(usize),
    #[cfg(feature = "tetration-rocm")]
    Rocm(usize),
    #[cfg(all(feature = "tetration-metal", target_os = "macos"))]
    Metal,
}

#[cfg(any(
    feature = "tetration-gpu",
    feature = "tetration-metal",
    feature = "tetration-rocm"
))]
impl StreamingGpuBackend {
    fn vram_check_chunk(&self, bytes: usize) -> Result<(), &'static str> {
        match self {
            #[cfg(feature = "tetration-gpu")]
            Self::Cuda(idx) => cuda::vram_check(*idx, bytes),
            #[cfg(feature = "tetration-rocm")]
            Self::Rocm(idx) => rocm::vram_check(*idx, bytes),
            #[cfg(all(feature = "tetration-metal", target_os = "macos"))]
            Self::Metal => metal::vram_check(bytes),
        }
    }

    fn reduce_chunk(
        &self,
        vals: &[f32],
        kind: ReductionKind,
    ) -> Result<ScalarReductionResult, &'static str> {
        match self {
            #[cfg(feature = "tetration-gpu")]
            Self::Cuda(idx) => {
                cuda::reduce_f32_scalar(vals, kind, *idx).map_err(|_| "gpu_runtime_error")
            }
            #[cfg(feature = "tetration-rocm")]
            Self::Rocm(idx) => {
                rocm::reduce_f32_scalar(vals, kind, *idx).map_err(|_| "gpu_runtime_error")
            }
            #[cfg(all(feature = "tetration-metal", target_os = "macos"))]
            Self::Metal => metal::reduce_f32_scalar(vals, kind).map_err(|_| "gpu_runtime_error"),
        }
    }
}

#[cfg(any(
    feature = "tetration-gpu",
    feature = "tetration-metal",
    feature = "tetration-rocm"
))]
pub(crate) fn try_streaming_f32_fold(
    backend: StreamingGpuBackend,
    used: &'static str,
    cuda_device: Option<usize>,
    mmap: &[u8],
    plan: &ReadPlan,
    max_preview: usize,
    kind: ReductionKind,
    route: DeviceRoute,
    f16_input: bool,
) -> Result<(FoldPlanOutcome, DeviceRoute), &'static str> {
    streaming_fold_chunk_subset(
        backend,
        used,
        cuda_device,
        mmap,
        plan,
        max_preview,
        kind,
        route,
        f16_input,
        None,
        plan.chunks.len() > 1,
    )
}

/// Fold a subset of planned chunks (or all when `chunk_indices` is `None`).
#[cfg(any(
    feature = "tetration-gpu",
    feature = "tetration-metal",
    feature = "tetration-rocm"
))]
pub(crate) fn streaming_fold_chunk_subset(
    backend: StreamingGpuBackend,
    used: &'static str,
    cuda_device: Option<usize>,
    mmap: &[u8],
    plan: &ReadPlan,
    max_preview: usize,
    kind: ReductionKind,
    route: DeviceRoute,
    f16_input: bool,
    chunk_indices: Option<&[usize]>,
    pipeline: bool,
) -> Result<(FoldPlanOutcome, DeviceRoute), &'static str> {
    if matches!(kind, ReductionKind::Var | ReductionKind::Std) {
        return streaming_cpu_value_fold(
            mmap,
            plan,
            max_preview,
            kind,
            route,
            used,
            cuda_device,
            f16_input,
            chunk_indices,
        );
    }

    let indices: Vec<usize> = match chunk_indices {
        Some(ix) => ix.to_vec(),
        None => (0..plan.chunks.len()).collect(),
    };

    let n = plan.logical_f32_element_count;
    let preview_cap = max_preview.min(n);
    let mut preview = vec![0.0_f32; preview_cap];
    let mut gpu_acc = ScalarReductionResult::default_fields(0);
    let mut cpu_value = ValueAccum::default();
    let mut cpu_arg = ArgIndexAccum::default();
    let mut total_bytes_read_from_disk = 0u64;

    let use_pipeline = pipeline && indices.len() > 1;
    if use_pipeline {
        fold_chunks_pipelined(
            mmap,
            plan,
            &indices,
            preview_cap,
            &mut preview,
            kind,
            &backend,
            f16_input,
            &mut gpu_acc,
            &mut cpu_value,
            &mut cpu_arg,
            &mut total_bytes_read_from_disk,
        )?;
    } else {
        fold_chunks_sequential(
            mmap,
            plan,
            &indices,
            preview_cap,
            &mut preview,
            kind,
            &backend,
            f16_input,
            &mut gpu_acc,
            &mut cpu_value,
            &mut cpu_arg,
            &mut total_bytes_read_from_disk,
        )?;
    }

    if gpu_acc.element_count == 0 && cpu_value.is_empty() {
        return Err("gpu_empty_selection");
    }

    let operation = if gpu_acc.element_count > 0 {
        gpu_acc.finalize_merged(kind).into()
    } else {
        cpu_value.finish_scalar(kind).into()
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
        gpu_pipeline: use_pipeline,
        gpu_multi: route.gpu_multi,
        cuda_device,
    };
    Ok((outcome, ok_route))
}

#[cfg(any(
    feature = "tetration-gpu",
    feature = "tetration-metal",
    feature = "tetration-rocm"
))]
struct ChunkPayload {
    bytes: u64,
    vals: Vec<f32>,
}

#[cfg(any(
    feature = "tetration-gpu",
    feature = "tetration-metal",
    feature = "tetration-rocm"
))]
fn fold_chunks_sequential(
    mmap: &[u8],
    plan: &ReadPlan,
    indices: &[usize],
    preview_cap: usize,
    preview: &mut [f32],
    kind: ReductionKind,
    backend: &StreamingGpuBackend,
    f16_input: bool,
    gpu_acc: &mut ScalarReductionResult,
    cpu_value: &mut ValueAccum,
    cpu_arg: &mut ArgIndexAccum,
    total_bytes: &mut u64,
) -> Result<(), &'static str> {
    for &i in indices {
        let c = &plan.chunks[i];
        let (chunk_bytes, vals) =
            collect_planned_chunk_values(mmap, plan, c, preview_cap, preview, f16_input)
                .map_err(|_| "gpu_host_decode_failed")?;
        *total_bytes = total_bytes
            .checked_add(chunk_bytes)
            .ok_or("gpu_host_decode_failed")?;
        reduce_chunk_payload(
            vals,
            chunk_bytes,
            kind,
            backend,
            gpu_acc,
            cpu_value,
            cpu_arg,
        )?;
    }
    Ok(())
}

#[cfg(any(
    feature = "tetration-gpu",
    feature = "tetration-metal",
    feature = "tetration-rocm"
))]
fn fold_chunks_pipelined(
    mmap: &[u8],
    plan: &ReadPlan,
    indices: &[usize],
    preview_cap: usize,
    preview: &mut [f32],
    kind: ReductionKind,
    backend: &StreamingGpuBackend,
    f16_input: bool,
    gpu_acc: &mut ScalarReductionResult,
    cpu_value: &mut ValueAccum,
    cpu_arg: &mut ArgIndexAccum,
    total_bytes: &mut u64,
) -> Result<(), &'static str> {
    let (work_tx, work_rx) = mpsc::sync_channel::<Result<ChunkPayload, TetError>>(2);
    let plan_chunks = plan.chunks.clone();
    let plan_geom = (
        plan.dataset_shape.clone(),
        plan.chunk_shape.clone(),
        plan.selection_box_start.clone(),
        plan.selection_box_stop_exclusive.clone(),
        plan.selection_step.clone(),
        plan.logical_selection_shape.clone(),
        plan.logical_f32_element_count,
        plan.chunk_touch_policy,
    );

    let decode_indices = indices.to_vec();
    let mmap_vec = mmap.to_vec();
    let decode_handle = thread::spawn(move || {
        let plan = rebuild_plan_for_decode(&plan_chunks, plan_geom);
        for &i in &decode_indices {
            let c = plan.chunks[i].clone();
            let mut local_preview = vec![0.0_f32; preview_cap];
            let payload = collect_planned_chunk_values(
                &mmap_vec,
                &plan,
                &c,
                preview_cap,
                &mut local_preview,
                f16_input,
            )
            .map(|(bytes, vals)| ChunkPayload { bytes, vals });
            if work_tx.send(payload).is_err() {
                break;
            }
        }
    });

    let mut preview_filled = false;
    for payload in work_rx {
        let ChunkPayload { bytes, vals } = payload.map_err(|_| "gpu_host_decode_failed")?;
        if !preview_filled && preview_cap > 0 && !vals.is_empty() {
            for (i, &v) in vals.iter().take(preview_cap).enumerate() {
                preview[i] = v;
            }
            preview_filled = true;
        }
        *total_bytes = total_bytes
            .checked_add(bytes)
            .ok_or("gpu_host_decode_failed")?;
        reduce_chunk_payload(vals, bytes, kind, backend, gpu_acc, cpu_value, cpu_arg)?;
    }
    decode_handle.join().map_err(|_| "gpu_runtime_error")?;
    Ok(())
}

#[cfg(any(
    feature = "tetration-gpu",
    feature = "tetration-metal",
    feature = "tetration-rocm"
))]
fn rebuild_plan_for_decode(
    chunks: &[PlannedChunkIo],
    geom: (
        Vec<u64>,
        Vec<u64>,
        Vec<u64>,
        Vec<u64>,
        Vec<u64>,
        Vec<u64>,
        usize,
        &'static str,
    ),
) -> ReadPlan {
    let (
        dataset_shape,
        chunk_shape,
        selection_box_start,
        selection_box_stop_exclusive,
        selection_step,
        logical_selection_shape,
        logical_f32_element_count,
        chunk_touch_policy,
    ) = geom;
    ReadPlan {
        chunk_touch_policy,
        chunk_count: chunks.len(),
        total_stored_bytes: chunks.iter().map(|c| c.stored_byte_len).sum(),
        chunks: chunks.to_vec(),
        dataset_shape,
        chunk_shape,
        selection_box_start,
        selection_box_stop_exclusive,
        selection_step,
        logical_selection_shape,
        logical_f32_element_count,
    }
}

#[cfg(any(
    feature = "tetration-gpu",
    feature = "tetration-metal",
    feature = "tetration-rocm"
))]
fn reduce_chunk_payload(
    vals: Vec<f32>,
    _chunk_bytes: u64,
    kind: ReductionKind,
    backend: &StreamingGpuBackend,
    gpu_acc: &mut ScalarReductionResult,
    cpu_value: &mut ValueAccum,
    cpu_arg: &mut ArgIndexAccum,
) -> Result<(), &'static str> {
    if vals.is_empty() {
        return Ok(());
    }
    let chunk_bytes_usize = vals
        .len()
        .checked_mul(4)
        .ok_or("gpu_logical_bytes_overflow")?;
    if backend.vram_check_chunk(chunk_bytes_usize).is_ok() {
        let part = backend.reduce_chunk(&vals, kind)?;
        gpu_acc.merge_partial(&part, kind);
    } else {
        fold_values_into_cpu_accum(&vals, kind, cpu_value, cpu_arg);
    }
    Ok(())
}

#[cfg(any(
    feature = "tetration-gpu",
    feature = "tetration-metal",
    feature = "tetration-rocm"
))]
fn streaming_cpu_value_fold(
    mmap: &[u8],
    plan: &ReadPlan,
    max_preview: usize,
    kind: ReductionKind,
    route: DeviceRoute,
    used: &'static str,
    cuda_device: Option<usize>,
    f16_input: bool,
    chunk_indices: Option<&[usize]>,
) -> Result<(FoldPlanOutcome, DeviceRoute), &'static str> {
    let indices: Vec<usize> = match chunk_indices {
        Some(ix) => ix.to_vec(),
        None => (0..plan.chunks.len()).collect(),
    };
    let n = plan.logical_f32_element_count;
    let preview_cap = max_preview.min(n);
    let mut preview = vec![0.0_f32; preview_cap];
    let mut value = ValueAccum::default();
    let mut arg = ArgIndexAccum::default();
    let mut total_bytes_read_from_disk = 0u64;

    for &i in &indices {
        let c = &plan.chunks[i];
        let (chunk_bytes, vals) =
            collect_planned_chunk_values(mmap, plan, c, preview_cap, &mut preview, f16_input)
                .map_err(|_| "gpu_host_decode_failed")?;
        total_bytes_read_from_disk = total_bytes_read_from_disk
            .checked_add(chunk_bytes)
            .ok_or("gpu_host_decode_failed")?;
        if matches!(kind, ReductionKind::Var | ReductionKind::Std) {
            value.push_f32_le_bytes(bytemuck::cast_slice(&vals), ReductionKind::Var);
        } else {
            fold_values_into_cpu_accum(&vals, kind, &mut value, &mut arg);
        }
    }

    if value.is_empty() {
        return Err("gpu_empty_selection");
    }

    fold::shared::validate_fold_preview(n > 0, &preview, preview_cap)
        .map_err(|_| "gpu_empty_selection")?;

    let operation = value.finish_scalar(kind).into();
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
        gpu_pipeline: false,
        gpu_multi: route.gpu_multi,
        cuda_device,
    };
    Ok((outcome, ok_route))
}

#[cfg(any(
    feature = "tetration-gpu",
    feature = "tetration-metal",
    feature = "tetration-rocm"
))]
pub(crate) fn collect_planned_chunk_values(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    preview_cap: usize,
    preview: &mut [f32],
    f16_input: bool,
) -> Result<(u64, Vec<f32>), TetError> {
    let mut vals = Vec::new();
    let bytes = if f16_input {
        visit_planned_chunk_f16(mmap, plan, c, |li, v| {
            let f = f32::from(v);
            if li < preview_cap {
                preview[li] = f;
            }
            vals.push(f);
            Ok(())
        })?
    } else {
        visit_planned_chunk(mmap, plan, c, |li, v| {
            if li < preview_cap {
                preview[li] = v;
            }
            vals.push(v);
            Ok(())
        })?
    };
    Ok((bytes, vals))
}

#[cfg(any(
    feature = "tetration-gpu",
    feature = "tetration-metal",
    feature = "tetration-rocm"
))]
fn fold_values_into_cpu_accum(
    vals: &[f32],
    kind: ReductionKind,
    value: &mut ValueAccum,
    arg: &mut ArgIndexAccum,
) {
    let _ = arg;
    for &v in vals {
        match kind {
            ReductionKind::ArgMin | ReductionKind::ArgMax => {
                unreachable!("gpu streaming does not support argmin/argmax")
            }
            _ => value.push(v),
        }
    }
}

/// Partial fold over a chunk index subset (no preview), for multi-GPU merge.
#[cfg(any(feature = "tetration-gpu", feature = "tetration-rocm"))]
pub(crate) fn streaming_fold_partial_driver(
    backend: StreamingGpuBackend,
    mmap: &[u8],
    plan: &ReadPlan,
    kind: ReductionKind,
    f16_input: bool,
    chunk_indices: &[usize],
) -> Result<(ScalarReductionResult, u64), &'static str> {
    if matches!(kind, ReductionKind::Var | ReductionKind::Std) {
        let mut value = ValueAccum::default();
        let mut total = 0u64;
        let mut preview: &mut [f32] = &mut [];
        for &i in chunk_indices {
            let c = &plan.chunks[i];
            let (bytes, vals) =
                collect_planned_chunk_values(mmap, plan, c, 0, &mut preview, f16_input)
                    .map_err(|_| "gpu_host_decode_failed")?;
            total = total.checked_add(bytes).ok_or("gpu_host_decode_failed")?;
            value.push_f32_le_bytes(bytemuck::cast_slice(&vals), ReductionKind::Var);
        }
        if value.is_empty() {
            return Ok((ScalarReductionResult::default_fields(0), total));
        }
        return Ok((value.finish_scalar(kind), total));
    }

    let mut gpu_acc = ScalarReductionResult::default_fields(0);
    let mut cpu_value = ValueAccum::default();
    let mut cpu_arg = ArgIndexAccum::default();
    let mut total = 0u64;
    let mut preview: &mut [f32] = &mut [];
    fold_chunks_sequential(
        mmap,
        plan,
        chunk_indices,
        0,
        &mut preview,
        kind,
        &backend,
        f16_input,
        &mut gpu_acc,
        &mut cpu_value,
        &mut cpu_arg,
        &mut total,
    )?;
    if gpu_acc.element_count == 0 && cpu_value.is_empty() {
        return Ok((ScalarReductionResult::default_fields(0), total));
    }
    let acc = if gpu_acc.element_count > 0 {
        gpu_acc.finalize_merged(kind)
    } else {
        cpu_value.finish_scalar(kind)
    };
    Ok((acc, total))
}
