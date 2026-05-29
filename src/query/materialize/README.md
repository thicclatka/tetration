# `query/materialize` — logical dense buffers

Decode planned chunks into **row-major logical tensors** (or spill files) for reductions, previews, transforms, and covariance.

## Entry points

Per-dtype materialize + spill:

- `materialize_read_plan_f32_le` (+ `_into`, `_parallel` variants)
- `f64`, `f16`, and integer paths via `int/`

## Files

| File / dir                   | Role                                                  |
| ---------------------------- | ----------------------------------------------------- |
| `f32.rs`, `f64.rs`, `f16.rs` | Float decode + scatter                                |
| `int/`                       | `u8`–`i64` materialize and integer folds              |
| `logical.rs`                 | `MaterializedLogical` — type-erased backing for stats |
| `selection.rs`               | Build logical view from materialized or spill file    |
| `parallel.rs`                | Rayon chunk workers into shared buffer                |
| `shared.rs`                  | Common decode loop helpers                            |
| `validate.rs`                | Geometry checks before allocate                       |
| `stats.rs`                   | Tier-C statistics materialization                     |
| `covariance.rs`              | Pairwise covariance / correlation matrices            |
| `types.rs`                   | `DecodePreviewBundle`, backing enums                  |

## Outcomes

- **In-core:** `MaterializeReadPlanF32IntoOutcome` with preview samples
- **Spill:** `spill_read_plan_*` writes raw logical bytes to `WriteHints` path
- **Fold without full dense RAM:** delegates to `fold/` via `dispatch.rs`

## Related

- Budget choice: `engine/budget.rs`
- After materialize: `fold/reduction`, `transform/apply`
