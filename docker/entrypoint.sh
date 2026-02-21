#!/usr/bin/env bash
set -euo pipefail

python --version || true

# If user passes a command, run it.
if [[ $# -gt 0 ]]; then
  exec "$@"
fi

# Default: open shell
exec bash