# `ffi` — C ABI (feature `tetration-ffi`)

Stable C entrypoints for embedders who cannot use Rust directly. Built as `cdylib` + `rlib`; headers in [`include/tetration.h`](../../include/tetration.h).

Enable: `cargo build --features tetration-ffi`

Docs: [`docs/ffi.md`](../../docs/ffi.md)

## Symbols

| C API                                | Rust backing                                |
| ------------------------------------ | ------------------------------------------- |
| `tet_abi_version`                    | `TET_ABI_VERSION`                           |
| `tet_open` / `tet_close`             | `TetFile` boxed handle                      |
| `tet_summary_json`                   | `TetFile::summary()` → JSON                 |
| `tet_query_json`                     | `execute_query_json` → `QueryResponse` JSON |
| `tet_verify_json`                    | `verify_tet_bytes` (no handle)              |
| `tet_last_error` / `tet_clear_error` | Thread-local error string                   |
| `tet_string_free`                    | Free JSON return buffers                    |

## Files

| File       | Role                                       |
| ---------- | ------------------------------------------ |
| `mod.rs`   | `#[no_mangle]` exports, panic guard        |
| `error.rs` | Thread-local `last_error` for failed calls |

## Contract

- Returned `char*` JSON must be freed with `tet_string_free`
- Handles are opaque; do not use after `tet_close`
- ABI version bumps on breaking C layout/symbol changes

## Example

[`examples/ffi_query.c`](../../examples/ffi_query.c)
