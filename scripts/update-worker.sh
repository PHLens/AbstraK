#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/activate-volume-tools.sh"

REMOTE_URL="${ABSTRAK_REMOTE_URL:-https://github.com/PHLens/AbstraK.git}"
BRANCH="${ABSTRAK_BRANCH:-main}"

command -v git >/dev/null || { echo "git is required in PATH" >&2; exit 1; }
test -d "$ABSTRAK_ROOT/.git" || { echo "missing AbstraK checkout" >&2; exit 1; }

current_branch="$(git -C "$ABSTRAK_ROOT" symbolic-ref --quiet --short HEAD)" || {
  echo "AbstraK checkout must be on branch $BRANCH" >&2
  exit 1
}
test "$current_branch" = "$BRANCH" || {
  echo "AbstraK checkout is on $current_branch, expected $BRANCH" >&2
  exit 1
}
test -z "$(git -C "$ABSTRAK_ROOT" status --porcelain=v1)" || {
  echo "AbstraK checkout is dirty; preserve or commit changes before updating" >&2
  exit 1
}

git -C "$ABSTRAK_ROOT" remote set-url origin "$REMOTE_URL"
git -C "$ABSTRAK_ROOT" pull --ff-only origin "$BRANCH"
