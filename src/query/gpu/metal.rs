//! Apple Metal scalar `f32` reductions (macOS only).

use std::sync::OnceLock;

use metal::{CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize};

use crate::query::fold::reduction::{ReductionKind, ScalarReductionResult};
use crate::query::gpu::{self, GPU_VRAM_BUFFER_FRACTION};

const BLOCK_THREADS: u64 = 256;
const MSL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

#define BLOCK_SIZE 256

kernel void block_reduce_sum(
    device const float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint grid_dim [[threadgroups_per_grid]]
) {
    threadgroup float sdata[BLOCK_SIZE];
    float acc = 0.f;
    uint stride = BLOCK_SIZE * grid_dim;
    for (uint i = bid * BLOCK_SIZE + tid; i < n; i += stride) {
        acc += in[i];
    }
    sdata[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        out[bid] = sdata[0];
    }
}

kernel void block_reduce_min(
    device const float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint grid_dim [[threadgroups_per_grid]]
) {
    threadgroup float sdata[BLOCK_SIZE];
    float acc = numeric_limits<float>::max();
    uint stride = BLOCK_SIZE * grid_dim;
    for (uint i = bid * BLOCK_SIZE + tid; i < n; i += stride) {
        acc = min(acc, in[i]);
    }
    sdata[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] = min(sdata[tid], sdata[tid + offset]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        out[bid] = sdata[0];
    }
}

kernel void block_reduce_max(
    device const float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint grid_dim [[threadgroups_per_grid]]
) {
    threadgroup float sdata[BLOCK_SIZE];
    float acc = -numeric_limits<float>::max();
    uint stride = BLOCK_SIZE * grid_dim;
    for (uint i = bid * BLOCK_SIZE + tid; i < n; i += stride) {
        acc = max(acc, in[i]);
    }
    sdata[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] = max(sdata[tid], sdata[tid + offset]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        out[bid] = sdata[0];
    }
}
"#;

struct MetalKernels {
    sum: ComputePipelineState,
    min: ComputePipelineState,
    max: ComputePipelineState,
}

struct MetalState {
    device: Device,
    queue: metal::CommandQueue,
    kernels: MetalKernels,
}

static METAL_STATE: OnceLock<Result<MetalState, String>> = OnceLock::new();

fn metal_state() -> Result<&'static MetalState, &'static str> {
    METAL_STATE
        .get_or_init(init_metal)
        .as_ref()
        .map_err(|_| "gpu_runtime_error")
}

fn init_metal() -> Result<MetalState, String> {
    let device = Device::system_default().ok_or_else(|| "gpu_no_device".to_string())?;
    let queue = device.new_command_queue();
    let library = device
        .new_library_with_source(MSL_SRC, &CompileOptions::new())
        .map_err(|e| e.to_string())?;
    let kernels = MetalKernels {
        sum: pipeline(&device, &library, "block_reduce_sum")?,
        min: pipeline(&device, &library, "block_reduce_min")?,
        max: pipeline(&device, &library, "block_reduce_max")?,
    };
    Ok(MetalState {
        device,
        queue,
        kernels,
    })
}

fn pipeline(
    device: &Device,
    library: &metal::Library,
    name: &str,
) -> Result<ComputePipelineState, String> {
    let function = library
        .get_function(name, None)
        .map_err(|e| e.to_string())?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| e.to_string())
}

/// Best-effort unified-memory budget check before host→GPU upload.
pub(crate) fn vram_check(logical_bytes: usize) -> Result<(), &'static str> {
    let state = metal_state()?;
    let budget = state.device.recommended_max_working_set_size();
    let limit = ((budget as f64) * GPU_VRAM_BUFFER_FRACTION) as u64;
    if u64::try_from(logical_bytes).ok().is_some_and(|b| b > limit) {
        return Err("gpu_vram_exceeded");
    }
    Ok(())
}

pub(crate) fn reduce_f32_scalar(
    host: &[f32],
    kind: ReductionKind,
) -> Result<ScalarReductionResult, String> {
    let n = host.len();
    if n == 0 {
        return Ok(ScalarReductionResult {
            element_count: 0,
            ..ScalarReductionResult::default_fields(0)
        });
    }
    if matches!(kind, ReductionKind::Var | ReductionKind::Std) {
        let var = gpu::host_f32_population_variance(host);
        return Ok(ScalarReductionResult {
            element_count: n,
            var_scalar: Some(var),
            std_scalar: if matches!(kind, ReductionKind::Std) {
                Some(var.sqrt())
            } else {
                None
            },
            ..ScalarReductionResult::default_fields(n)
        });
    }

    let state = metal_state().map_err(|e| e.to_string())?;
    let in_buf = state.device.new_buffer_with_data(
        host.as_ptr().cast(),
        u64::try_from(std::mem::size_of_val(host)).map_err(|_| "gpu_logical_bytes_overflow")?,
        MTLResourceOptions::StorageModeShared,
    );

    match kind {
        ReductionKind::Count => Ok(ScalarReductionResult {
            element_count: n,
            ..ScalarReductionResult::default_fields(n)
        }),
        ReductionKind::Sum | ReductionKind::Mean => {
            let sum = block_tree_reduce_f32(state, &state.kernels.sum, &in_buf, n)?;
            let sum_f64 = f64::from(sum);
            Ok(ScalarReductionResult {
                element_count: n,
                sum_scalar: Some(sum_f64),
                mean_scalar: if matches!(kind, ReductionKind::Mean) {
                    Some(sum_f64 / n as f64)
                } else {
                    None
                },
                ..ScalarReductionResult::default_fields(n)
            })
        }
        ReductionKind::Min => {
            let v = block_tree_reduce_f32(state, &state.kernels.min, &in_buf, n)?;
            Ok(ScalarReductionResult {
                element_count: n,
                min_scalar: Some(f64::from(v)),
                ..ScalarReductionResult::default_fields(n)
            })
        }
        ReductionKind::Max => {
            let v = block_tree_reduce_f32(state, &state.kernels.max, &in_buf, n)?;
            Ok(ScalarReductionResult {
                element_count: n,
                max_scalar: Some(f64::from(v)),
                ..ScalarReductionResult::default_fields(n)
            })
        }
        _ => Err("gpu_unsupported_op".into()),
    }
}

fn block_tree_reduce_f32(
    state: &MetalState,
    pipeline: &ComputePipelineState,
    input: &metal::Buffer,
    n: usize,
) -> Result<f32, String> {
    let mut levels: Vec<metal::Buffer> = Vec::new();
    let mut current: &metal::Buffer = input;
    let mut len = n;
    loop {
        if len == 0 {
            return Ok(0.0);
        }
        if len == 1 {
            let ptr = current.contents() as *const f32;
            return Ok(unsafe { *ptr });
        }
        let grid = grid_dim(len);
        let partials = state.device.new_buffer(
            (grid * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        launch_block_reduce(state, pipeline, current, &partials, len)?;
        levels.push(partials);
        current = levels.last().expect("level buffer");
        len = grid;
    }
}

fn grid_dim(n: usize) -> usize {
    n.div_ceil(BLOCK_THREADS as usize).max(1)
}

fn launch_block_reduce(
    state: &MetalState,
    pipeline: &ComputePipelineState,
    input: &metal::Buffer,
    output: &metal::Buffer,
    n: usize,
) -> Result<(), String> {
    let grid = grid_dim(n);
    let n_u32 = u32::try_from(n).map_err(|_| "gpu_length_overflow".to_string())?;
    let n_buf = state.device.new_buffer_with_data(
        &n_u32 as *const u32 as *const std::ffi::c_void,
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = state.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(input), 0);
    encoder.set_buffer(1, Some(output), 0);
    encoder.set_buffer(2, Some(&n_buf), 0);
    let thread_group_size = MTLSize::new(BLOCK_THREADS, 1, 1);
    let thread_groups = MTLSize::new(grid as u64, 1, 1);
    encoder.dispatch_thread_groups(thread_groups, thread_group_size);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    Ok(())
}
