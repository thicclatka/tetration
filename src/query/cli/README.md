# `query/cli` — CLI presentation layer

Formatting helpers for the **`tet`** binary (`query`, `info`, `qhist` output). Not required for library embedders, but several formatters are re-exported at `query::*` for convenience.

## Submodules

| Path         | Role                                                         |
| ------------ | ------------------------------------------------------------ |
| `info.rs`    | `tet info` — dataset table, JSON, quiet summary              |
| `history.rs` | `tet qhist` — platform query history (`query_history.jsonl`) |
| `text.rs`    | Shared ASCII helpers                                         |
| `output/`    | Query response formatters (see below)                        |

## `output/` formatters

| File            | `--format`                                   |
| --------------- | -------------------------------------------- |
| `quiet.rs`      | `quiet` / `-q` — one-line stdout             |
| `stats.rs`      | `stats` — slim JSON aggregates               |
| `table.rs`      | `table` — ASCII tables + slice grid          |
| `plan.rs`       | `plan` — catalog + read_plan only            |
| `format_num.rs` | Number formatting for tables                 |
| `hints.rs`      | stderr hints (spill paths, warnings)         |
| `mod.rs`        | `QueryOutputFormat`, `format_query_response` |

## Query history

Stored outside the `.tet` file:

- Default path from `cli_query_history_path()`
- Env: `TET_NO_QUERY_HISTORY`, `TET_QUERY_HISTORY_FILE`, `TET_QUERY_HISTORY_MAX`

## Related

- Binary wiring: [`bin/tet/query.rs`](../../bin/tet/query.rs), `info.rs`, `qhist.rs`
- Core execution unchanged — CLI calls `execute_query_json` when `-x` is set
