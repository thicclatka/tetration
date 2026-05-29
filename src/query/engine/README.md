# `query/engine` — plan + execute orchestration

Glue between planning, materialization, folding, transforms, and spill export. Public planning functions are re-exported at the `query` crate root.

## Files

| File              | Role                                                                    |
| ----------------- | ----------------------------------------------------------------------- |
| `run.rs`          | `plan_query_with_tet_mmap_ex`, main execute path                        |
| `operations.rs`   | Match `Operation` variant → fold / materialize / transform / covariance |
| `budget.rs`       | `ExecutionBudget`, `MemoryStrategy` (in-core vs spill vs streaming)     |
| `spill_policy.rs` | Temp spill files, path allowlist (`--spill-allow`)                      |

## Memory strategies

| Strategy                                                  | When                             |
| --------------------------------------------------------- | -------------------------------- |
| `streaming_fold`                                          | Tier-A/B `operation`             |
| `in_memory_materialize` / `temp_spill_materialize`        | Tier-C ops                       |
| `mmap_spill`                                              | Top-level `spill` export         |
| `transform_ram` / `transform_spill` / `transform_sidecar` | `transform` with `write` routing |
| `capped_in_memory`                                        | Preview-only execute             |

See [`docs/query_engine.md`](../../../docs/query_engine.md#memory-budget-and-execution-strategies).

## Key exports

- `plan_query_empty`, `plan_query_with_tet_mmap`, `plan_query_with_tet_mmap_ex`
- `materialize_read_plan_*` (re-exported from `materialize`)
- `SpillPathAllowlist`, `ExecutionBudget`, `DEFAULT_MEMORY_BUDGET_BYTES`

## Callers

- `execute.rs` — embedder API
- `bin/tet/query.rs` — CLI `-x`
- `ffi` — `tet_query_json`
