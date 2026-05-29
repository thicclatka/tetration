# `query/transform` — two-pass element-wise transforms

Implements the `transform` query operation: **pass 1** collects statistics over the selection; **pass 2** decodes again and rewrites each element (in RAM or via spill).

## Files

| File          | Role                                                                     |
| ------------- | ------------------------------------------------------------------------ |
| `mod.rs`      | `run_transform`, `TransformRunInput`, orchestration                      |
| `stats.rs`    | Pass 1 — mean/std/min/max/etc. per `TransformMethod`                     |
| `apply.rs`    | Pass 2 — zscore, minmax, l1/l2 norm, center, scale, log1p, sqrt, softmax |
| `target.rs`   | Output dtype / write target resolution                                   |
| `warnings.rs` | `TransformWarnings` (div-by-zero, invalid sqrt → NaN)                    |

## Supported methods

See `types/transform_method.rs` — `TransformMethod` wire strings match query JSON keys.

## Flow

```text
Pass 1: fold/stats over logical selection
Pass 2: decode chunks → apply f(element, stats) → preview or spill path
```

Division by zero or invalid `sqrt` shift yields **NaN** and records a warning in the execution preview.

## Related

- Uses `engine` budget + spill policy
- Dtype paths via `dispatch.rs`
- CLI preview: `cli/output/quiet.rs`, `stats.rs`
