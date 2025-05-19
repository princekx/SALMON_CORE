#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --mem=40gb
#SBATCH --cpus-per-task=8

# RUN AS sbatch ./my_sbatch.sh yyyy-mm-dd
# Check if a date argument is provided (format: YYYY-MM-DD)
if [[ $1 =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    date="$1"
else
    date=$(date -d "yesterday" +%F)
fi

echo "Using date: $date"
# Run Eq waves
/home/users/prince.xavier/MJO/SALMON/salmon_run.py -d $date -t 00 -a eqwaves -m mogreps
/home/users/prince.xavier/MJO/SALMON/salmon_run.py -d $date -t 06 -a eqwaves -m mogreps
/home/users/prince.xavier/MJO/SALMON/salmon_run.py -d $date -t 12 -a eqwaves -m mogreps
/home/users/prince.xavier/MJO/SALMON/salmon_run.py -d $date -t 18 -a eqwaves -m mogreps
