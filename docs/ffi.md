# C ABI (Phase 11)

Stable **`extern "C"`** entry points for non-Rust runtimes. The Rust crate (`tetration`) remains the full API; the shared library exposes a **narrow** subset aligned with CLI parity where it matters.

## Python

Official Python bindings are **not** built from this repository. A separate PyPI project (repository **TBD**) will ship wheels (PyO3 / maturin) that depend on a pinned **`tetration`** release on [crates.io](https://crates.io/crates/tetration). Use that repo for NumPy integration, packaging, and Python-specific convert paths (`h5py`, `zarr`, etc.).

Until then: shell **`tet`**, embed **Rust**, or implement readers from [`layout_v1.md`](layout_v1.md).

## Status

**Not implemented yet.** Planned surface is documented in [`GETTING_STARTED.md` — Phase 11](../GETTING_STARTED.md#phase-11--c-abi-cdylib).

## Design principles (v1)

1. **JSON for documents** — query in and structured result out match `tet query -x` / `execute_query_json` + serde `QueryResponse`, so bindings do not marshal Rust structs by hand.
2. **Opaque handles** — `TetHandle` owns mmap + path; no exposed Rust types across the FFI edge.
3. **Caller-owned buffers** — functions that return JSON allocate with `tet_string_free` (name TBD); no Rust `String` pointers with hidden allocators.
4. **Lean library** — `tetration-ffi` builds with **`default-features = false`** (no HDF5, NetCDF, GPU in `libtetration`).
5. **ABI versioning** — `TET_ABI_VERSION` integer in the header; bump on breaking C layout changes (independent of crate semver patch).

## Planned symbols (illustrative)

| Symbol                   | Role                              |
| ------------------------ | --------------------------------- |
| `tet_abi_version`        | Compile-time / runtime ABI check  |
| `tet_open` / `tet_close` | Open `.tet` read-only             |
| `tet_last_error`         | UTF-8 error after failed call     |
| `tet_summary_json`       | File + dataset catalog summary    |
| `tet_query_json`         | Execute query document JSON       |
| `tet_string_free`        | Free buffers returned by `*_json` |

**Out of v1:** convert/import, writer session, GPU device selection, query history.

## Stability

- **0.x crate:** C ABI may change between minors until `1.0`; document in release notes.
- **Panics:** FFI entry points use `catch_unwind` or `panic = "abort"` for the cdylib build; document which.

## See also

- Embedder Rust API: [`GETTING_STARTED.md` — Rust API by phase](../GETTING_STARTED.md#rust-api-by-phase)
- Query JSON wire: [`query_engine.md`](query_engine.md)
- Layout for standalone readers: [`layout_v1.md`](layout_v1.md)
