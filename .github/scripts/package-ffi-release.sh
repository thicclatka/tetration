#!/usr/bin/env bash
# Package lean libtetration + include/tetration.h for GitHub Releases.
# Env: TAG (e.g. v0.1.6), PLATFORM (e.g. linux-x86_64, macos-aarch64, windows-x86_64).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

TAG="${TAG:?TAG is required (e.g. v0.1.6)}"
PLATFORM="${PLATFORM:?PLATFORM is required (e.g. linux-x86_64)}"

cargo build --release --no-default-features --features tetration-ffi

OUT="target/release"
STAGING="dist/tetration-ffi-${TAG}-${PLATFORM}"
mkdir -p "$STAGING/include" "$STAGING/lib"

cp include/tetration.h "$STAGING/include/"

case "$(uname -s)" in
  Linux)
    cp "$OUT/libtetration.so" "$STAGING/lib/"
    ;;
  Darwin)
    cp "$OUT/libtetration.dylib" "$STAGING/lib/"
    ;;
  MINGW* | MSYS* | CYGWIN* | Windows*)
  if [[ -f "$OUT/tetration.dll" ]]; then
      cp "$OUT/tetration.dll" "$STAGING/lib/"
    else
      echo "package-ffi-release: missing $OUT/tetration.dll" >&2
      exit 1
    fi
    ;;
  *)
    echo "package-ffi-release: unsupported OS $(uname -s)" >&2
    exit 1
    ;;
esac

cat >"$STAGING/README.txt" <<EOF
Tetration C ABI (${TAG}, ${PLATFORM})
=====================================

  include/tetration.h   — C declarations (TET_ABI_VERSION)
  lib/                  — shared library (built with --features tetration-ffi, no HDF5/NetCDF)

Link (Unix example):
  cc -I include your.c -L lib -ltetration

See https://github.com/Latka-Industries/tetration/blob/main/docs/ffi.md
EOF

mkdir -p dist
ARCHIVE="dist/tetration-ffi-${TAG}-${PLATFORM}.tar.gz"
tar -C dist -czf "$ARCHIVE" "$(basename "$STAGING")"
echo "package-ffi-release: wrote $ARCHIVE"
