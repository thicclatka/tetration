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

```text
Estimate logical bytes + budget (file header bps or default)
    → InCore: parallel materialize + fold
    → SpillExport: write logical selection to user path
    → StreamingFold: chunk-at-a-time without dense RAM (CPU or GPU)
```

## Key exports

- `plan_query_empty`, `plan_query_with_tet_mmap`, `plan_query_with_tet_mmap_ex`
- `materialize_read_plan_*` (re-exported from `materialize`)
- `SpillPathAllowlist`, `ExecutionBudget`, `DEFAULT_MEMORY_BUDGET_BYTES`

## Callers

- `execute.rs` — embedder API
- `bin/tet/query.rs` — CLI `-x`
- `ffi` — `tet_query_json`
