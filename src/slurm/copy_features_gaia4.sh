#!/usr/bin/env bash
#SBATCH --partition=robolab
#SBATCH --nodelist=gaia4
#SBATCH --job-name=copy_features_from_gaia4
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=log/slurm/copy_features_from_gaia4.%j.out
#SBATCH --error=log/slurm/copy_features_from_gaia4.%j.err

set -euo pipefail

for dir in $(ls -d /data/henicke/*Satclip*40*Features); do
    srun rsync -avh --progress "${dir}/" "gaia5:${dir}/" 
    srun rsync -avh --progress "${dir}/" "gaia7:${dir}/" 
done