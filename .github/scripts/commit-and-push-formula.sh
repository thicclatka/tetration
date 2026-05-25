#!/usr/bin/env bash
set -euo pipefail

# Env: TAG (required). Optional: FORMULA=Formula/tetration.rb, BRANCH=main

: "${TAG:?TAG must be set}"

FORMULA="${FORMULA:-Formula/tetration.rb}"
BRANCH="${BRANCH:-main}"

git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

git add "$FORMULA"

if git diff --cached --quiet; then
  echo "No formula changes (already up to date)."
  exit 0
fi

git commit -m "brew: bump formula to ${TAG}"
git pull --rebase origin "$BRANCH"
git push origin "$BRANCH"
