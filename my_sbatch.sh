#!/bin/bash
#SBATCH --time=5:00
#SBATCH --mem=40gb
#SBATCH --cpus-per-task=8


date="2025-04-04"

#date=$(date -d "yesterday" +%F)
echo $date
# Run MJO
./salmon_run.py -d $date -a mjo -m mogreps
sleep 60

# Run ColdSurge
./salmon_run.py -d $date  -a coldsurge -m mogreps

sleep 60
# Run Eq waves
./salmon_run.py -d $date -t 00 -a eqwaves -m mogreps
./salmon_run.py -d $date -t 06 -a eqwaves -m mogreps
./salmon_run.py -d $date -t 12 -a eqwaves -m mogreps
./salmon_run.py -d $date -t 18 -a eqwaves -m mogreps

sleep 60
