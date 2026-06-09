#!/bin/bash
set -euxo pipefail
# Get the recipe name from the first argument
RECIPE=$1
# Use yesterday's date in UTC.
DATE=$(date -u -d "yesterday" +%Y-%m-%d)

# Resolve project root from the installed workflow source symlink.
SOURCE_CYLC_DIR=$(readlink -f "${CYLC_WORKFLOW_RUN_DIR}/../_cylc-install/source")
PROJECT_ROOT=$(cd "${SOURCE_CYLC_DIR}/.." && pwd)
RECIPE_PATH="${PROJECT_ROOT}/recipes/${RECIPE}.yaml"

if [[ ! -f "${RECIPE_PATH}" ]]; then
	echo "Recipe not found: ${RECIPE_PATH}" >&2
	exit 1
fi

# Run the recipe using an absolute recipe path.
salmon run "${RECIPE_PATH}" --date "$DATE"
