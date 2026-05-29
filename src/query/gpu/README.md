# `query/gpu` — optional device reductions (Phase 10)

Experimental GPU path for **tier-A/B scalar `f32`** (and **`f16`** on device) when `execution.device` requests CUDA, ROCm, or Metal. CPU streaming fold remains the default for large selections.

## Cargo features (mutually exclusive NVIDIA vs AMD)

| Feature           | Backend                                       |
| ----------------- | --------------------------------------------- |
| `tetration-gpu`   | CUDA (`cuda`, `cuda:multi`, `auto` on NVIDIA) |
| `tetration-rocm`  | HIP via cudarc (`rocm`, `rocm:multi`)         |
| `tetration-metal` | Metal on macOS (`metal`, `auto`)              |

Do **not** enable `tetration-gpu` and `tetration-rocm` together.

## Files

| File                | Role                                              |
| ------------------- | ------------------------------------------------- |
| `scalar_fold.rs`    | Dense buffer → device kernel → scalar result      |
| `streaming_fold.rs` | Chunk stream to device when host RAM insufficient |
| `multi.rs`          | Shard chunks across multiple GPUs                 |
| `cuda.rs`           | NVIDIA driver / NVRTC                             |
| `rocm.rs`           | AMD HIP                                           |
| `metal.rs`          | Apple GPU                                         |

## Routing

`query/device.rs` resolves `ExecutionDeviceHint` → `DeviceRoute`, checks backend availability, and decides host materialize vs streaming based on `GPU_HOST_MATERIALIZE_RAM_FRACTION`.

## Related

- Host decode still uses `decode/` + `catalog` codecs
- Multi-GPU: same chunk-parallel shape as CPU `fold/parallel`
- Docs: [`docs/query_engine.md`](../../docs/query_engine.md) Phase 10 section
