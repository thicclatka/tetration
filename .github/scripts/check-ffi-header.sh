#!/usr/bin/env bash
# Ensure include/tetration.h matches #[no_mangle] exports in src/ffi/mod.rs.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

HEADER="include/tetration.h"
RUST="src/ffi/mod.rs"

rust_syms="$(mktemp)"
hdr_syms="$(mktemp)"
trap 'rm -f "$rust_syms" "$hdr_syms"' EXIT

grep -E 'pub unsafe extern "C" fn (tet_[a-z_]+)' "$RUST" | sed -E 's/.*fn (tet_[a-z_]+)\(.*/\1/' | sort -u >"$rust_syms"
grep -E '^\w.+\btet_[a-z_]+\(' "$HEADER" | grep -oE 'tet_[a-z_]+' | sort -u >"$hdr_syms"

if ! diff -u "$rust_syms" "$hdr_syms"; then
  echo "check-ffi-header: symbol list mismatch between $HEADER and $RUST" >&2
  exit 1
fi

header_ver="$(sed -n 's/^#define TET_ABI_VERSION \([0-9]*\)u/\1/p' "$HEADER")"
rust_ver="$(sed -n 's/^pub const TET_ABI_VERSION: u32 = \([0-9]*\);/\1/p' "$RUST")"

if [[ -z "$header_ver" || -z "$rust_ver" ]]; then
  echo "check-ffi-header: could not parse TET_ABI_VERSION from header or Rust" >&2
  exit 1
fi

if [[ "$header_ver" != "$rust_ver" ]]; then
  echo "check-ffi-header: TET_ABI_VERSION mismatch (header=$header_ver rust=$rust_ver)" >&2
  exit 1
fi

count="$(wc -l <"$rust_syms" | tr -d ' ')"
echo "check-ffi-header: ok ($count symbols, ABI version $rust_ver)"
