#!/bin/bash
# Restores the TimescaleDB dump into the fresh container on first init.
# Runs LAST (999-) so the timescaledb extension is already installed.
# Wrapped in pre/post_restore as TimescaleDB requires for logical dumps.
set -e

if [ ! -f /dump/index.sql ]; then
  echo ">> No /dump/index.sql found — starting with an EMPTY database."
  echo ">> Analytical + prediction paths will say 'I'm not sure' until data is loaded."
  exit 0
fi

echo ">> Restoring TimescaleDB dump — this can take a few minutes..."
psql -v ON_ERROR_STOP=0 -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
     -c "SELECT timescaledb_pre_restore();"
psql -v ON_ERROR_STOP=0 -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
     -f /dump/index.sql
psql -v ON_ERROR_STOP=0 -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
     -c "SELECT timescaledb_post_restore();"
echo ">> Restore complete."
