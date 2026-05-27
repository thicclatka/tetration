# Phase 9 task list (`phase9/query-ops`)

Branch: **`phase9/query-ops`** → merge to `main` when complete. Phase 6 (TOML query, preview table) follows on a separate branch.

| #   | Task                                                                   | Status        |
| --- | ---------------------------------------------------------------------- | ------------- |
| 1   | **Named axes** — `"mean": "time"` via footer `dim_names`               | done          |
| 2   | **Histogram** — caller `min` / `max` bin edges                         | done          |
| 3   | **QC counts** — `nan_count`, `null_count` (fill-aware)                 | done          |
| 4   | **Covariance / correlation** — tier C along axis                       | done (rank-2) |
| 5   | **Coordinate selection** — `start_label` / `stop_label` on `selection` | done          |
| 6   | **Export** — `.tet` → Zarr directory                                   | done          |
| 7   | **`inf_count`** — ±inf element count (tier A/B)                        | done          |
| 8   | Docs + tests + `GETTING_STARTED` checkboxes                            | done          |

**Out of scope (Phase 9):** layout v2, FFT/ML ops, SQL/joins.

**Verify each slice:** `cargo test --lib`, `cargo clippy -- -D warnings`.
