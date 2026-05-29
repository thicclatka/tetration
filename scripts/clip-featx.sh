#!/usr/bin/env bash
# Run `cargo clippy -D warnings` across optional feature sets (mirrors CI matrix).
#
# Usage (from repo root):
#   scripts/clip-featx.sh
#   mise run clip-featx

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

CLIPPY_ARGS=(-- -D warnings)

ci_features() {
  local feat="tetration-netcdf,tetration-hdf5,tetration-gpu"
  case "$(uname -s)" in
    Darwin) feat="${feat},tetration-metal" ;;
  esac
  printf '%s' "$feat"
}

run_clip() {
  local label="$1"
  shift
  echo ""
  echo "==> cargo clippy ${label}"
  cargo clippy "$@" "${CLIPPY_ARGS[@]}"
}

run_clip "(no features)" --no-default-features

for feat in tetration-ffi tetration-netcdf tetration-hdf5 tetration-gpu tetration-rocm; do
  run_clip "--no-default-features --features ${feat}" --no-default-features --features "$feat"
done

if [[ "$(uname -s)" == Darwin ]]; then
  run_clip "--no-default-features --features tetration-metal" \
    --no-default-features --features tetration-metal
else
  echo ""
  echo "==> skip tetration-metal (macOS only)"
fi

run_clip "(default features)"

ci_feat="$(ci_features)"
run_clip "--features ${ci_feat} (CI bundle)" --features "$ci_feat"
run_clip "--features ${ci_feat},tetration-ffi (CI + FFI)" --features "${ci_feat},tetration-ffi"

echo ""
echo "clip-featx: all clippy passes OK"
