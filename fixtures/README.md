# Fixtures

Python generators and checked-in tensors for **convert**, **query**, and **memory-stress** work. See [`GETTING_STARTED.md`](../GETTING_STARTED.md) for the full phase checklist.

## By phase

| Phase | Role of fixtures |
| ----- | ---------------- |
| **1–3** | Tests build temp `.tet` in `tests/fixture.rs`; no tracked import fixtures yet. |
| **4** | Query tests use programmatic `.tet` files; optional manual runs against converted outputs. |
| **5** | **This directory** — HDF5 / NetCDF small roundtrips (`tests/convert.rs`); large 20 GiB stress (local only). |
| **5 (next)** | Planned: **Zarr** directory store under `small/zarr/`; **groups / CF** variants alongside existing tensors. |
| **6** | Python binding tests may reuse `small/` sources; convert path uses Python libs, not Rust HDF5/NetCDF in wheels. |
| **7** | History footer today (`convert` events); future fixture attrs preserved into `.tet` dataset metadata on import. |

## Layout

| Path                         | Git         | Contents                                                                                 |
| ---------------------------- | ----------- | ---------------------------------------------------------------------------------------- |
| `small/h5/`, `small/netcdf/` | **Tracked** | `tensor_3d`, `tensor_4d`, `tensor_5d` — seeded arrays with **f32, f64, i32, i64** vars   |
| `large/h5/`, `large/netcdf/` | **Ignored** | `tensor_20gb.h5`, `tensor_20gb.nc` — **20 GiB** logical f32 each (slab-written, low RAM) |

Each small file holds four variables named by dtype: `f32`, `f64`, `i32`, `i64`.

| File        | Shape        | Logical size (per dtype)             |
| ----------- | ------------ | ------------------------------------ |
| `tensor_3d` | 32 × 32 × 32 | 128 KiB (f32/i32), 256 KiB (f64/i64) |
| `tensor_4d` | 16⁴          | 256 KiB (f32/i32), 512 KiB (f64/i64) |
| `tensor_5d` | 8⁵           | 128 KiB (f32/i32), 256 KiB (f64/i64) |

Float arrays use a mild `linspace` + noise pattern; integer arrays use a deterministic ramp with small jitter.

File attributes: `tetration_fixture`, `tetration_ndim`, `tetration_dtypes`. Each variable also carries `tetration_dtype`.

Large files keep a single `data` variable (`float32` only) with `tetration_logical_bytes`.

## Tests

| Consumer | What it checks |
| -------- | -------------- |
| `tests/convert.rs` | `tet convert` on every `small/` tensor × dtype; byte equality vs source; parallel `--jobs 4` smoke |
| Manual / bench | `fixtures/large/*` — throughput and peak RAM (`tet convert … --jobs 0`) |

Regenerate tracked small files after changing `generate.py` shapes or dtypes, then re-run `cargo test --test convert`.

## Regenerate

From the repo root (via [mise](https://mise.jdx.dev/)):

```bash
mise run fixtures:small        # 6 small files (default)
mise run fixtures:large        # two 20 GiB files; needs ~40 GiB disk
mise run fixtures:all
mise run fixtures:clean-large  # remove fixtures/large/ (prompts for confirmation)
```

Or from this directory with uv directly:

```bash
uv sync
uv run generate-fixtures small    # 6 small files (default)
uv run generate-fixtures large    # two 20 GiB files; needs ~40 GiB disk
uv run generate-fixtures all
uv run generate-fixtures large -q   # no tqdm / status lines
```

Progress: **tqdm** bars per file (large writes show ~320 slabs × 64 MiB). Or: `uv run python generate.py large`

Large files are written in **~64 MiB** slabs so peak memory stays modest; the run still needs **~40 GiB free disk** and time proportional to I/O.

## Planned (Phase 5)

- **`small/zarr/`** — v2 (or v3) directory stores mirroring `tensor_3d` / `4d` / `5d` shapes for Zarr → `.tet` tests.
- **Richer HDF5/NetCDF** — optional `small/h5/groups/`, CF attrs (`scale_factor`, `_FillValue`), multi-group layouts; extend `generate.py` when importers land.

Other dense formats (**`.npy`**, GRIB, GeoTIFF, …) get fixtures only if/when convert support is added — see Phase 5 “could add later” in [`GETTING_STARTED.md`](../GETTING_STARTED.md).
