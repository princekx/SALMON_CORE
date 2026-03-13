#!/usr/bin/env bash
# ============================================================
# SALMON dev_run.sh  —  Run salmon run during development
#
# Usage:
#   ./dev_run.sh                        # uses today's date, mjo_mogreps recipe
#   ./dev_run.sh --recipe coldsurge     # use recipes/coldsurge.yaml
#   ./dev_run.sh --date 2024-01-15      # use a specific date
#   ./dev_run.sh --no-debug             # suppress debug logging
#   ./dev_run.sh --recipe mjo_mogreps --date 2024-06-01
#
# Because salmon is installed in editable mode (pip install -e .),
# every code change you make is picked up immediately on the next run.
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPES_DIR="$SCRIPT_DIR/recipes"

# ── Defaults ────────────────────────────────────────────────
RECIPE="mjo_mogreps"
DATE="$(date +%Y-%m-%d)"
DEBUG="--debug"

# ── Parse arguments ─────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --recipe)  RECIPE="$2";     shift 2 ;;
        --date)    DATE="$2";       shift 2 ;;
        --no-debug) DEBUG="";       shift   ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

RECIPE_FILE="$RECIPES_DIR/${RECIPE}.yaml"

# ── Validate ────────────────────────────────────────────────
if [[ ! -f "$RECIPE_FILE" ]]; then
    echo "ERROR: Recipe not found: $RECIPE_FILE"
    echo "Available recipes:"
    ls "$RECIPES_DIR"/*.yaml | xargs -n1 basename
    exit 1
fi

# ── Run ─────────────────────────────────────────────────────
echo "======================================================"
echo "  SALMON Development Run"
echo "  Recipe : $RECIPE_FILE"
echo "  Date   : $DATE"
echo "  Debug  : ${DEBUG:-(off)}"
echo "======================================================"
echo ""

salmon run "$RECIPE_FILE" --date "$DATE" $DEBUG
