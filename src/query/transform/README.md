# `query/transform` — two-pass element-wise transforms

Implements the `transform` query operation: **pass 1** collects statistics over the selection; **pass 2** decodes again and rewrites each element (in RAM, spill bytes, or a published sidecar `.tet`).

## Files

| File          | Role                                                                                               |
| ------------- | -------------------------------------------------------------------------------------------------- |
| `mod.rs`      | `run_transform`, `TransformRunInput`, `materialize_transform_dense_ram`, orchestration             |
| `stats.rs`    | Pass 1 — mean/std/min/max/etc. per `TransformMethod`                                               |
| `apply.rs`    | Pass 2 — zscore, minmax, l1/l2 norm, center, scale, log1p, sqrt, softmax                           |
| `target.rs`   | `write` target resolution (`ram`, `spill`, `switch`, `sidecar`)                                    |
| `sidecar.rs`  | Draft `.tet` in cache → `publish_file` beside source; footer history + `{source}-{method}` dataset |
| `warnings.rs` | `TransformWarnings` (div-by-zero, invalid sqrt → NaN)                                              |

## Supported methods

See `types/transform_method.rs` — `TransformMethod` wire strings match query JSON keys.

## `write` targets

| Target    | Memory strategy                      | Output                                                                    |
| --------- | ------------------------------------ | ------------------------------------------------------------------------- |
| `switch`  | `transform_ram` or `transform_spill` | RAM when selection ≤ budget; else spill file (default)                    |
| `ram`     | `transform_ram`                      | Dense buffer in RAM; preview from transformed values                      |
| `spill`   | `transform_spill`                    | Dtype-native bytes at caller path or cache temp                           |
| `sidecar` | `transform_sidecar`                  | One-chunk `.tet` beside source (`write.path`, `write.timestamp` optional) |

Sidecar requires a source `.tet` path (`--tet` / `tet_path`). Pass-2 materializes the full logical selection in RAM before writing the sidecar file.

Embedders that need a full transformed array without publishing a `.tet` should use [`materialize_query_transform_ram`](../embed_materialize.rs) (`write: ram` only).

## Flow

```text
Pass 1: fold/stats over logical selection
Pass 2: decode chunks → apply f(element, stats) → preview, spill path, or sidecar .tet
```

Division by zero or invalid `sqrt` shift yields **NaN** and records a warning in the execution preview.

## Related

- Uses `engine` budget + spill policy
- Dtype paths via `dispatch.rs`
- CLI preview: `cli/output/quiet.rs`, `stats.rs`
- Docs: [`docs/query_engine.md`](../../../docs/query_engine.md#embedder-dense-export)
