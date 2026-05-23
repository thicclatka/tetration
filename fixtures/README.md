# Fixtures (HDF5 / NetCDF)

Python generators for **convert** and memory stress tests.

## Layout

| Path                         | Git         | Contents                                                                                 |
| ---------------------------- | ----------- | ---------------------------------------------------------------------------------------- |
| `small/h5/`, `small/netcdf/` | **Tracked** | `tensor_3d`, `tensor_4d`, `tensor_5d` — seeded arrays with **f32, f64, i32, i64** vars   |
| `large/h5/`, `large/netcdf/` | **Ignored** | `tensor_20gb.h5`, `tensor_20gb.nc` — **20 GiB** logical f32 each (slab-written, low RAM) |

## Regenerate

From this directory:

```bash
uv sync
uv run generate-fixtures small    # 6 small files (default)
uv run generate-fixtures large    # two 20 GiB files; needs ~40 GiB disk
uv run generate-fixtures all
uv run generate-fixtures large -q   # no tqdm / status lines
```

Progress: **tqdm** bars per file (large writes show ~320 slabs × 64 MiB). Or: `uv run python generate.py large`

Large files are written in **~64 MiB** slabs so peak memory stays modest; the run still needs **~40 GiB free disk** and time proportional to I/O.

## Small shapes

Each file holds four variables named by dtype: `f32`, `f64`, `i32`, `i64`.

| File        | Shape        | Logical size (per dtype)             |
| ----------- | ------------ | ------------------------------------ |
| `tensor_3d` | 32 × 32 × 32 | 128 KiB (f32/i32), 256 KiB (f64/i64) |
| `tensor_4d` | 16⁴          | 256 KiB (f32/i32), 512 KiB (f64/i64) |
| `tensor_5d` | 8⁵           | 128 KiB (f32/i32), 256 KiB (f64/i64) |

Float arrays use a mild `linspace` + noise pattern; integer arrays use a deterministic ramp with small jitter.

File attributes: `tetration_fixture`, `tetration_ndim`, `tetration_dtypes`. Each variable also carries `tetration_dtype`.

Large files keep a single `data` variable (`float32` only) with `tetration_logical_bytes`.
