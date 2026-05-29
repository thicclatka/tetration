# `tests/` — integration tests

`#[cfg(test)]` module from [`lib.rs`](../lib.rs). Exercises end-to-end behavior across catalog, query, convert, verify, and CLI — not part of the public API.

## Layout

| File                                                       | Focus                                   |
| ---------------------------------------------------------- | --------------------------------------- |
| `fixture.rs`, `query_fixtures.rs`, `small_tet_fixtures.rs` | Shared `.tet` bytes and query documents |
| `catalog.rs`, `session.rs`, `metadata.rs`                  | Writers, readers, footer metadata       |
| `query.rs`, `fold.rs`, `reduction.rs`, `covariance.rs`     | Query execution and aggregates          |
| `resolve_axes.rs`, `resolve_selection.rs`                  | Selection resolution                    |
| `convert.rs`, `export.rs`                                  | Foreign format round trips              |
| `verify.rs`, `verify_fixtures.rs`, `repair.rs`             | Health checks and repair                |
| `cli_output.rs`, `cli_info.rs`, `cli_history.rs`           | CLI formatters                          |
| `concurrent_query.rs`, `device.rs`, `fs_device.rs`         | Concurrency and device routing          |
| `ffi.rs`                                                   | C ABI smoke (feature-gated)             |
| `layout_roundtrip.rs`, `utils.rs`, `variance_simd.rs`      | Layout and SIMD unit coverage           |

## Running

```bash
cargo test
cargo test query::          # query module tests
cargo test --features tetration-ffi ffi  # FFI tests
```

Fixtures also live under repo [`fixtures/`](../../fixtures/).
