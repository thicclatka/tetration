# Small `.tet` fixtures (verify / repair / query)

Tracked binaries for **manual** CLI checks and `src/tests/small_tet_fixtures.rs`.

| File | Role |
| ---- | ---- |
| `sample.tet` | Multichunk `f32` `[2,3]` / tiles `[2,2]` (values 1..6) |
| `large.tet` | `f32` `[34,64]` / `[4,4]` → **144** chunks (quick verify samples 128) |
| `plan.tet` | Same as `sample` + **invalid** history footer (`tet repair` plan target) |
| `multichunk_u8.tet` | Multichunk `u8`, values 1..6 |
| `multichunk_u32.tet` | Multichunk `u32`, values 1..6 |
| `multichunk_f16.tet` | Multichunk `f16`, values 1..6 |

## Manual (from repo root)

```bash
cargo build --release
TET=./target/release/tet

$TET verify fixtures/small/tet/sample.tet
$TET verify fixtures/small/tet/large.tet          # ok; warns about skipped decode past chunk 128
$TET verify --deep fixtures/small/tet/large.tet -q
$TET repair fixtures/small/tet/plan.tet            # plan (footer_invalid)
$TET repair fixtures/small/tet/plan.tet --apply footer_invalid

$TET query '{"dataset":"a","sum":[]}' -t fixtures/small/tet/multichunk_u8.tet -x -q
$TET query '{"dataset":"a","var":[]}' -t fixtures/small/tet/multichunk_u8.tet -x -q
$TET query '{"dataset":"a","sum":[]}' -t fixtures/small/tet/multichunk_u32.tet -x -q
$TET query '{"dataset":"a","var":[]}' -t fixtures/small/tet/multichunk_f16.tet -x -q
```

Expected aggregates on 1..6: **sum=21**, **var≈2.916667** (population, `ddof=0`).

## Regenerate

```bash
UPDATE_SMALL_TET=1 cargo test --lib regenerate_tracked_small_tet_fixtures -- --ignored --nocapture
```

Then commit `fixtures/small/tet/*.tet`.
