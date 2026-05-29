# `query/fold` — reductions and partial aggregates

Chunk-local and merged **scalar reductions** (mean, sum, min, max, variance, nan\_\* variants) and **partial-axis** folds without materializing the full tensor when possible.

## Submodule map

| Path                  | Role                                                      |
| --------------------- | --------------------------------------------------------- |
| `fold_policy.rs`      | `FoldIoPolicy` — in-core parallel vs linear scan vs spill |
| `linear_scan.rs`      | Out-of-core: one chunk at a time on CPU                   |
| `parallel/`           | Rayon workers, per-chunk partials, merge                  |
| `partial/`            | Partial reductions along subset of axes                   |
| `partial_geometry.rs` | Geometry for partial-axis fold layouts                    |
| `reduction/`          | `ReductionKind`, Welford variance, scalar accumulators    |
| `shared.rs`           | `FoldPlanOutcome` and shared helpers                      |
| `variance_simd/`      | SIMD fast paths for float/integer variance                |

## `reduction/` detail

| File             | Role                                |
| ---------------- | ----------------------------------- |
| `scalar.rs`      | min/max/sum/mean-style scalar folds |
| `welford.rs`     | Online variance / std               |
| `value_accum.rs` | Generic accumulators                |

## `parallel/` detail

| File         | Role                                      |
| ------------ | ----------------------------------------- |
| `workers.rs` | Thread pool chunk assignment              |
| `partial.rs` | Per-chunk partial state                   |
| `merge.rs`   | Combine partials (associative reductions) |
| `scalar.rs`  | Parallel scalar fold driver               |
| `preview.rs` | Preview sample collection during fold     |

## Execution tiers (informal)

- **Tier A/B:** scalar reductions over selection (this module)
- **Tier C:** higher-order stats (`materialize/stats.rs`)
- **Partial:** collapse some axes, keep others (`partial/`)

## Related

- Dtype dispatch: `query/dispatch.rs`
- GPU scalar fold when device selected: `gpu/scalar_fold.rs`
- Transform pass-1 stats reuse fold machinery: `transform/stats.rs`
