#!/usr/bin/env python3
"""Set Formula `url` and `sha256` from the GitHub tag archive (TAG + REPO env)."""

from __future__ import annotations

import hashlib
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def main() -> None:
    tag = os.environ.get("TAG", "").strip()
    repo = os.environ.get("REPO", "").strip()
    formula = Path(os.environ.get("FORMULA", "Formula/tetration.rb"))

    if not tag or not repo:
        print("TAG and REPO must be set (e.g. from github.ref_name / github.repository)", file=sys.stderr)
        sys.exit(1)

    url = f"https://github.com/{repo}/archive/refs/tags/{tag}.tar.gz"

    data = None
    last_err: BaseException | None = None
    for attempt in range(8):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                data = resp.read()
            break
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = e
            time.sleep(4)

    if data is None:
        print(f"Could not download {url}: {last_err!r}", file=sys.stderr)
        sys.exit(1)

    sha = hashlib.sha256(data).hexdigest()
    text = formula.read_text()
    text = re.sub(r'^  url ".*"$', f'  url "{url}"', text, flags=re.M)
    text = re.sub(r'^  sha256 ".*"$', f'  sha256 "{sha}"', text, flags=re.M)
    formula.write_text(text)

    print(f"url={url}")
    print(f"sha256={sha}")


if __name__ == "__main__":
    main()
