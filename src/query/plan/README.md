# `query/plan` — selection → read plan

Turns a validated `QueryDocument` + catalog into a **`ReadPlan`**: which chunks to touch and how each chunk maps into the logical selection.

## Files

| File           | Role                                                                       |
| -------------- | -------------------------------------------------------------------------- |
| `selection.rs` | Resolve `selection` / axis slices against dataset shape and coord metadata |
| `read_plan.rs` | Build `PlannedChunkIo` list, logical dims, byte estimates                  |

## Inputs

- `TetFileSummaryV1` or mmap + path (via `engine::plan_query_with_tet_mmap_ex`)
- `QueryDocument` with `dataset` and optional `selection`, `chunk_touch_policy`

## Outputs

`ReadPlan` — used by:

- Plan-only CLI (`tet query` without `-x`)
- Full execution (`engine::run`)
- GPU path (chunk list for streaming fold)

## Related

- Coord label resolution: `resolve_selection.rs`, `resolve_axes.rs` (parent `query/`)
- Chunk intersection math: `catalog::tile`
