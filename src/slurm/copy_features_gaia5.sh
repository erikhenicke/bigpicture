#!/usr/bin/env bash
#SBATCH --partition=robolab
#SBATCH --nodelist=gaia5
#SBATCH --job-name=copy_features_to_gaia5
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=log/slurm/copy_features_to_gaia5.%j.out
#SBATCH --error=log/slurm/copy_features_to_gaia5.%j.err

set -euo pipefail

for dir in $(ssh gaia7 'ls -d /data/henicke/*Satclip*Features'); do
    mkdir -p "${dir}"
    srun rsync -avh --progress "gaia7:${dir}/" "${dir}/"
done
