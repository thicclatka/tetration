#!/usr/bin/env bash
# Build examples/ffi_query.c against release libtetration and run on sample.tet.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SAMPLE="fixtures/small/tet/sample.tet"
if [[ ! -f "$SAMPLE" ]]; then
  echo "build-ffi-example: missing $SAMPLE" >&2
  exit 1
fi

cargo build --release --no-default-features --features tetration-ffi

OUT="target/release"
LIB_NAME="tetration"
EXE="$OUT/ffi_query"

cc -std=c11 -Wall -Wextra -Werror -I include examples/ffi_query.c -L "$OUT" -l"$LIB_NAME" -o "$EXE"

case "$(uname -s)" in
  Darwin)
    export DYLD_LIBRARY_PATH="$OUT${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
    ;;
  Linux)
    export LD_LIBRARY_PATH="$OUT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    ;;
  *)
    echo "build-ffi-example: unsupported OS for runtime test (built $EXE)" >&2
    exit 0
    ;;
esac

"$EXE" "$SAMPLE" | grep -q '"operation_mean":3.5' || {
  echo "build-ffi-example: expected operation_mean 3.5 on sample.tet" >&2
  exit 1
}

echo "build-ffi-example: ok ($EXE on $SAMPLE)"
