#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "source this script instead of executing it" >&2
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export ABSTRAK_ROOT="${ABSTRAK_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
export ABSTRAK_VOLUME_ROOT="${ABSTRAK_VOLUME_ROOT:-$(dirname -- "$ABSTRAK_ROOT")}"
export GIT_CONFIG_GLOBAL="${GIT_CONFIG_GLOBAL:-$ABSTRAK_VOLUME_ROOT/.gitconfig}"
export PATH="$ABSTRAK_VOLUME_ROOT/tools/bin:$PATH"

git_library_path="$ABSTRAK_VOLUME_ROOT/tools/lib"
if test -d "$git_library_path"; then
  export LD_LIBRARY_PATH="$git_library_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
