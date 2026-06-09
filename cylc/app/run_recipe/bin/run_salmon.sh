#!/bin/bash
set -euxo pipefail
RECIPE=$(echo $CYLC_TASK_NAME | sed 's/run_recipe_//')
bash /home/users/prince.xavier/MJO/SALMON_v2/SALMON_CORE/cylc/app/run_recipe/bin/run_recipe.sh $RECIPE
