# `repair` — in-place `.tet` fixes

Mutates files based on [`verify`](../verify/README.md) recommendations. **Verify stays read-only**; all writes happen here (or `tet verify --repair`).

## Public API

- `repair_tet_file` — plan or `--apply` specific codes
- `repair_plan` / `repair_plan_from_verify`
- `RepairOptions` — `dry_run`, `apply`, `plan_codes`
- `TetRepairReport` — per-action results + optional re-verify
- `is_repairable_code`, `repair_command_for_code`, `enrich_verify_recommendations`

## Submodules

| File         | Role                                          |
| ------------ | --------------------------------------------- |
| `plan.rs`    | Build `RepairPlan` from verify report         |
| `actions.rs` | Apply one code (e.g. strip bad `THST` footer) |
| `format.rs`  | JSON/text output for CLI                      |

## Supported repair codes (v1)

| Code             | Action                                   |
| ---------------- | ---------------------------------------- |
| `footer_invalid` | Remove or rewrite invalid history footer |

Other findings suggest re-convert or manual rewrite.

## CLI flow

```text
tet verify data.tet              → report + repair hints
tet repair data.tet               → plan (default)
tet repair data.tet --apply footer_invalid --dry-run
tet verify data.tet --repair      → verify then apply safe fixes
```
