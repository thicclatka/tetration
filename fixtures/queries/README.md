# Query document fixtures (JSON + TOML)

Shared query profiles for **`tet query`**, library parse tests, and CLI format smoke. Each stem has **`.json`** and **`.toml`** with the same semantics.

| Stem                       | Typical `.tet`                         | Notes                                                     |
| -------------------------- | -------------------------------------- | --------------------------------------------------------- |
| `mean_temperature`         | `fixtures/small/tet/sample.tet`        | Scalar mean, dataset **`temperature`**                    |
| `mean_strided_temperature` | `sample.tet`                           | Mean + strided `selection`                                |
| `slice_full_temperature`   | `sample.tet`                           | Preview-only; full **2×3** (`--preview 6 --format table`) |
| `slice_2x2_temperature`    | `sample.tet`                           | Preview **2×2** sub-slice (`--preview 4 --format table`)  |
| `mean_a`                   | `fixtures/small/tet/multichunk_u8.tet` | Scalar mean, dataset **`a`**                              |
| `sum_a`                    | `multichunk_u8.tet`                    | Scalar sum → **21** on values 1..6                        |
| `sum_axis0_a`              | `multichunk_u8.tet`                    | Partial sum on axis 0                                     |
| `var_a`                    | `multichunk_u8.tet`                    | Scalar var                                                |
| `quantile_axis0_a`         | `multichunk_u8.tet`                    | Quantile on axis 0                                        |

## Examples

```bash
BIN=./target/release/tet
SAMPLE=fixtures/small/tet/sample.tet
U8=fixtures/small/tet/multichunk_u8.tet
Q=fixtures/queries

$BIN query $Q/mean_temperature.toml -t $SAMPLE -x -q
$BIN query $Q/slice_full_temperature.json -t $SAMPLE -x --preview 6 --format table
$BIN query $Q/sum_a.toml -t $U8 -x -q
```

Tests load these via [`src/tests/fixture.rs`](../../src/tests/fixture.rs) (`query_files::json` / `::toml`).
