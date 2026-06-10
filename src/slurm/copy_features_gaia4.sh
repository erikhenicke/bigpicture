#!/usr/bin/env bash
#SBATCH --partition=robolab
#SBATCH --nodelist=gaia4
#SBATCH --job-name=copy_features_to_gaia4
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=log/slurm/copy_features_to_gaia4.%j.out
#SBATCH --error=log/slurm/copy_features_to_gaia4.%j.err

set -euo pipefail

for dir in $(ssh gaia7 'ls -d /data/henicke/*Satclip*Features'); do
    mkdir -p "${dir}"
    srun rsync -avh --progress "gaia7:${dir}/" "${dir}/"
done

# for dir in $(ssh gaia5 'ls -d /data/henicke/*Features'); do
#     mkdir -p "${dir}"
#     srun rsync -avh --progress "gaia5:${dir}/" "${dir}/"
# done
