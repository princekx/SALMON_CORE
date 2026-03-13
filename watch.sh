#!/usr/bin/env bash
# ============================================================
# SALMON watch.sh  —  Auto re-run salmon on code changes
#
# Usage:
#   ./watch.sh                      # watches salmon/, re-runs mjo_mogreps
#   ./watch.sh --recipe coldsurge   # use a different recipe
#   ./watch.sh --date 2024-01-15    # fix the date
#
# Requires: pip install watchfiles
# Press Ctrl+C to stop.
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPES_DIR="$SCRIPT_DIR/recipes"
WATCH_DIR="$SCRIPT_DIR/salmon"

RECIPE="mjo_mogreps"
DATE="$(date +%Y-%m-%d)"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --recipe) RECIPE="$2"; shift 2 ;;
        --date)   DATE="$2";   shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

RECIPE_FILE="$RECIPES_DIR/${RECIPE}.yaml"

echo "Watching $WATCH_DIR for changes..."
echo "Re-running: salmon run $RECIPE_FILE --date $DATE --debug"
echo "Press Ctrl+C to stop."
echo ""

# Run once immediately, then watch
run_salmon() {
    echo ""
    echo "──────────────────────────────────────────────"
    echo "  $(date '+%H:%M:%S') | Change detected — re-running..."
    echo "──────────────────────────────────────────────"
    salmon run "$RECIPE_FILE" --date "$DATE" --debug 2>&1 || true
}

run_salmon

# Use watchfiles to monitor the salmon source directory
python -c "
import subprocess, sys
from watchfiles import watch

watch_dir = '$WATCH_DIR'
recipe_file = '$RECIPE_FILE'
date = '$DATE'

print(f'Watching {watch_dir}...')
for changes in watch(watch_dir, watch_filter=lambda change, path: path.endswith('.py')):
    changed_files = [p for _, p in changes]
    print(f'Changed: {changed_files}')
    subprocess.run(['salmon', 'run', recipe_file, '--date', date, '--debug'])
"
