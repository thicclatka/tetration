# Fixtures

Python generators and checked-in tensors for **convert**, **query**, and **memory-stress** work. See [`GETTING_STARTED.md`](../GETTING_STARTED.md) for the full phase checklist.

## By phase

| Phase   | Role of fixtures                                                                                                                                                          |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1–3** | Tests build temp `.tet` in `src/tests/fixture.rs`; no tracked import fixtures yet.                                                                                        |
| **4**   | Query tests use programmatic `.tet` files; optional manual runs against converted outputs.                                                                                |
| **5**   | **This directory** — HDF5 / NetCDF / Zarr small roundtrips; large ~20 GiB **suite** split across three formats (local only).                                              |
| **6**   | Bench harness (`benchmark/`, `spec.json`); [`queries/`](queries/) JSON/TOML golden profiles; [`bench_results/latest.md`](bench_results/latest.md) for local perf reports. |
| **7**   | History footer today (`convert` events); future fixture attrs preserved into `.tet` dataset metadata on import.                                                           |
| **10**  | Python binding tests may reuse `small/` sources; convert path uses Python libs, not Rust HDF5/NetCDF in wheels.                                                           |

## Layout

### Small (tracked)

| Path                         | Contents                                                                                                                                                               |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `small/h5/`, `small/netcdf/` | **`tensor_{3,4,5}d`** — baseline root vars **f32, f64, i32, i64**                                                                                                      |
| `small/h5/`, `small/netcdf/` | **`groups_3d`** — nested **`primary/{dtype}`** + **`aux`**, **`meta`** groups                                                                                          |
| `small/h5/`, `small/netcdf/` | **`cf_3d`** — coords (**time/lat/lon**), **`temperature`** with **scale_factor / add_offset / \_FillValue**, plus root **f32…i64**                                     |
| `small/zarr/`                | **`tensor_{3,4,5}d`** directory stores; **`groups_3d`** with **`primary/`** subgroup                                                                                   |
| `small/tet/`                 | Tracked **`.tet`** for **`tet verify`**, **`tet verify --deep`**, **`tet repair`**, and **query** on **u8/u32/f16** — see [`small/tet/README.md`](small/tet/README.md) |
| `queries/`                   | Paired **JSON + TOML** query profiles for CLI smoke and `src/tests` — see [`queries/README.md`](queries/README.md)                                                     |

| File        | Shape              | Notes                                  |
| ----------- | ------------------ | -------------------------------------- |
| `tensor_3d` | 32³                | 128 KiB per f32/i32 var                |
| `tensor_4d` | 16⁴                | 256 KiB per f32/i32 var                |
| `tensor_5d` | 8⁵                 | 128 KiB per f32/i32 var                |
| `groups_3d` | 32³                | datasets under **`primary/`**          |
| `cf_3d`     | 32³ (time×lat×lon) | CF **`temperature`** + root dtype vars |

Float arrays use a mild `linspace` + noise pattern; integer arrays use a deterministic ramp with small jitter.

### Large (gitignored)

| Path                           | Size (logical f32)           |
| ------------------------------ | ---------------------------- |
| `large/h5/tensor_large.h5`     | ≈ **20 GiB ÷ 3** (~6.67 GiB) |
| `large/netcdf/tensor_large.nc` | ≈ **20 GiB ÷ 3**             |
| `large/zarr/tensor_large/`     | ≈ **20 GiB ÷ 3**             |

**Suite total ≈ 20 GiB** across HDF5 + NetCDF + Zarr. Slab writes use **~64 MiB** chunks so peak RAM stays modest during generation.

### Extra-large (gitignored)

One **20 GiB logical f32** file per format (original single-file stress layout):

| Path                                | Size       |
| ----------------------------------- | ---------- |
| `extra_large/h5/tensor_20gb.h5`     | **20 GiB** |
| `extra_large/netcdf/tensor_20gb.nc` | **20 GiB** |
| `extra_large/zarr/tensor_20gb/`     | **20 GiB** |

Generate only what you need (~20 GiB disk each):

```bash
mise run fixtures:extra-large-h5
mise run fixtures:extra-large-netcdf
mise run fixtures:extra-large-zarr
mise run fixtures:clean-extra-large
```

## Tests

| Consumer                          | What it checks                                                                                                                                       |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/tests/convert.rs`            | `tet convert` on **`tensor_*`**, **`groups_3d`**, **`cf_3d`**, and **Zarr** stores; byte equality vs source; parallel `--jobs 4` smoke; format sniff |
| `src/tests/small_tet_fixtures.rs` | Committed **`small/tet/*.tet`** — `tet verify` / `--deep` / `tet repair` / query sum\|var; see [`small/tet/README.md`](small/tet/README.md)          |
| `src/tests/query_fixtures.rs`     | **`fixtures/queries/`** JSON/TOML pairs parse to equivalent documents                                                                                |
| Manual / bench                    | `fixtures/large/*`, `fixtures/extra_large/*` — see [Benchmarks](#benchmarks); **`queries/`** + **`small/tet/`** for CLI/query smoke                  |

Regenerate tracked small files after changing the `generate/` package, then re-run `cargo test --lib tests::convert`.

## Benchmarks

Sequential per **tier**, then wipe the whole **format** tree before the next format.

1. Generate source
2. Source mean (native, chunked)
3. Convert → `.tet`
4. **Delete source** (only `.tet` needed from here)
5. `.tet` mean
6. **Delete `.tet`**
7. After both tiers (large + extra): **delete `large/{format}/` and `extra_large/{format}/`**

**Primary comparison:** full-tensor ops on the native file vs **`.tet`** query (mean, std, var, min, max, sum, count — see `fixtures/benchmark/spec.json`).  
**Secondary:** **convert** time (one-time import).

```bash
mise run bench              # h5, netcdf, zarr (large ~6.67 GiB + extra_large 20 GiB each)
mise run bench:h5           # one format only
uv run --directory fixtures bench-large --run-id my-run   # archived under bench_results/runs/
uv run --directory fixtures bench-large --skip-mean   # convert timing only
# or: uv run --directory fixtures tet-fixtures bench --skip-ops
```

Results (gitignored):

- `fixtures/bench_results/latest.md` — convenience copy of the last run
- `fixtures/bench_results/runs/<git>_<timestamp>/report.md` — archived markdown
- `fixtures/bench_results/runs/<git>_<timestamp>/report.json` — same run, machine-readable

Workload contract (committed, no blobs): `fixtures/benchmark/spec.json` — seeds, element counts, expected mean. Each case is verified after generate.

Report header lists CPU, RAM, query/convert workers (`--jobs 0` = auto), and GPU. Convert `--jobs 0` resolves to host parallelism (capped at 64), same as `tet convert`.

Peak disk per **extra_large** row ≈ **one** 20 GiB file (source or `.tet`, not both).

## Regenerate

From the repo root (via [mise](https://mise.jdx.dev/)):

```bash
mise run fixtures:small        # baseline + groups + cf + zarr (tracked)
mise run fixtures:large        # h5 + nc + zarr ≈ 20 GiB total; needs ~20 GiB disk
mise run fixtures:large-h5     # ~6.67 GiB HDF5 only
mise run fixtures:large-netcdf
mise run fixtures:large-zarr
mise run fixtures:extra-large-h5       # one 20 GiB HDF5 (~20 GiB disk)
mise run fixtures:extra-large-netcdf   # one 20 GiB NetCDF
mise run fixtures:extra-large-zarr     # one 20 GiB Zarr store
mise run fixtures:all
mise run fixtures:clean-large
mise run fixtures:clean-extra-large
```

Or from this directory with uv directly:

```bash
uv sync
uv run tet-fixtures generate small
uv run tet-fixtures generate large
uv run tet-fixtures generate extra-large-h5
uv run tet-fixtures generate extra-large-netcdf
uv run tet-fixtures generate extra-large-zarr
uv run tet-fixtures generate all
uv run tet-fixtures generate large -q

# legacy entry points still work:
uv run generate-fixtures small
uv run bench-large h5
```

Other dense formats (**`.npy`**, GRIB, GeoTIFF, …) get fixtures only if/when convert support is added — see Phase 5 “could add later” in [`GETTING_STARTED.md`](../GETTING_STARTED.md).
