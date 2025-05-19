#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --mem=40gb
#SBATCH --cpus-per-task=8

# RUN AS sbatch ./my_sbatch.sh
#date="2025-04-05"

date=$(date -d "yesterday" +%F)
#SBATCH --output=slurm_logs/slurm_${date}.out
echo $date
# Run MJO
./salmon_run.py -d $date -a mjo -m mogreps
sleep 30

# Run ColdSurge
./salmon_run.py -d $date  -a coldsurge -m mogreps

sleep 30
# Run Eq waves
./salmon_run.py -d $date -t 00 -a eqwaves -m mogreps
./salmon_run.py -d $date -t 06 -a eqwaves -m mogreps
./salmon_run.py -d $date -t 12 -a eqwaves -m mogreps
./salmon_run.py -d $date -t 18 -a eqwaves -m mogreps

sleep 30

# update JASMIN