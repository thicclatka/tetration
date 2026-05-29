#!/usr/bin/env bash
# Report filesystem device, mount point, and rotational (HDD) hint for one or more paths.
# Use before sidecar publish: if cache temp and destination .tet parent differ in DEV,
# the engine must copy+delete instead of rename.
#
# Usage:
#   scripts/fs-device-info.sh /path/to/data.tet ~/.cache/tetration
#   scripts/fs-device-info.sh /Volumes/bigdisk/foo.tet /tmp

set -euo pipefail

usage() {
  echo "Usage: $0 PATH [PATH...]" >&2
  exit 2
}

[[ $# -ge 1 ]] || usage

printf "%-8s %-6s %-12s %-8s %-10s %s\n" "DEV" "SAME?" "MOUNT" "FSTYPE" "ROT" "PATH"

first_dev=""
for path in "$@"; do
  if [[ ! -e "$path" ]]; then
    probe="$path"
    parent="$(dirname -- "$path")"
    [[ -d "$parent" ]] && probe="$parent"
  elif [[ -f "$path" ]]; then
    probe="$(dirname -- "$path")"
  else
    probe="$path"
  fi

  if [[ "$(uname -s)" == "Darwin" ]]; then
    dev="$(stat -f '%d' "$probe" 2>/dev/null || echo '?')"
    mount="$(df "$probe" 2>/dev/null | awk 'NR==2 {print $NF}')"
    fstype="$(df -T "$probe" 2>/dev/null | awk 'NR==2 {print $2}')"
    rot="n/a"
  else
    dev="$(stat -c '%d' "$probe" 2>/dev/null || echo '?')"
    mount="$(df --output=target "$probe" 2>/dev/null | tail -1 | tr -d ' ')"
    fstype="$(findmnt -n -o FSTYPE --target "$probe" 2>/dev/null || df -T "$probe" 2>/dev/null | awk 'NR==2 {print $2}')"
    rot="?"
    if [[ -n "$mount" && -r /sys/dev/block ]]; then
      majmin="$(stat -c '%t:%T' "$probe" 2>/dev/null || true)"
      if [[ -n "$majmin" ]]; then
        block="$(ls -l "/sys/dev/block/$majmin" 2>/dev/null | awk -F/ '{print $NF}' | sed 's/^[pv]//' | sed 's/[0-9]*$//')"
        if [[ -n "$block" && -r "/sys/block/$block/queue/rotational" ]]; then
          rot="$(cat "/sys/block/$block/queue/rotational" 2>/dev/null || echo '?')"
        fi
      fi
    fi
  fi

  same="—"
  if [[ -n "$first_dev" && "$dev" != "?" ]]; then
  if [[ "$dev" == "$first_dev" ]]; then same="yes"; else same="no"; fi
  elif [[ -z "$first_dev" && "$dev" != "?" ]]; then
    first_dev="$dev"
    same="ref"
  fi

  printf "%-8s %-6s %-12s %-8s %-10s %s\n" "$dev" "$same" "${mount:-?}" "${fstype:-?}" "$rot" "$path"
done

echo
echo "ROT: 1 ≈ HDD (rotational), 0 ≈ SSD/NVMe (Linux sysfs); macOS shows n/a."
echo "SAME?: rename works when DEV matches the first path (ref); copy+delete when no."
