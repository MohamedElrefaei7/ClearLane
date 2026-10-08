#!/bin/bash
# Weekly refresh: new 311 / Citi Bike data -> served artifact (no training, no test eval).
# Run by the LaunchAgent in scripts/com.clearlane.refresh.plist; safe to run by hand.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/logs
log="data/logs/refresh_$(date +%Y-%m-%d_%H%M).log"
exec >"$log" 2>&1
echo "refresh started $(date)"
.venv/bin/python -u -m clearlane.pipeline --refresh
echo "refresh finished $(date)"
