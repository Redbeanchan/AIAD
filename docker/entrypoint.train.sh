#!/usr/bin/env bash
set -euo pipefail

python --version || true

# Allow overriding command
if [[ $# -gt 0 ]]; then
  exec "$@"
fi

cd /app/background-removal

KEDRO_ENV="${KEDRO_ENV:-local}"
KEDRO_PIPELINE="${KEDRO_PIPELINE:-full}"

exec kedro run --env "${KEDRO_ENV}" --pipeline "${KEDRO_PIPELINE}"