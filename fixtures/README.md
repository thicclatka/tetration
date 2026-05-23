# Fixtures (HDF5 / NetCDF)

Python generators for **convert** and memory stress tests. Outputs use a single variable `data` (`float32`).

## Layout

| Path                         | Git         | Contents                                                                                 |
| ---------------------------- | ----------- | ---------------------------------------------------------------------------------------- |
| `small/h5/`, `small/netcdf/` | **Tracked** | `tensor_3d`, `tensor_4d`, `tensor_5d` — small seeded arrays                              |
| `large/h5/`, `large/netcdf/` | **Ignored** | `tensor_20gb.h5`, `tensor_20gb.nc` — **20 GiB** logical f32 each (slab-written, low RAM) |

## Regenerate

From this directory:

```bash
uv sync
uv run generate-fixtures small    # ~12 small files (default)
uv run generate-fixtures large    # two 20 GiB files; needs ~40 GiB disk
uv run generate-fixtures all
uv run generate-fixtures large -q   # no tqdm / status lines
```

Progress: **tqdm** bars per file (large writes show ~320 slabs × 64 MiB). Or: `uv run python generate.py large`

Large files are written in **~64 MiB** slabs so peak memory stays modest; the run still needs **~40 GiB free disk** and time proportional to I/O.

## Small shapes (f32)

| File        | Shape        | Logical size |
| ----------- | ------------ | ------------ |
| `tensor_3d` | 32 × 32 × 32 | 128 KiB      |
| `tensor_4d` | 16⁴          | 256 KiB      |
| `tensor_5d` | 8⁵           | 128 KiB      |

Attributes: `tetration_fixture`, `tetration_ndim` (small); `tetration_logical_bytes` (large).
