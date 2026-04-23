#!/usr/bin/env bash
#SBATCH --partition=robolab
#SBATCH --nodelist=gaia7
#SBATCH --job-name=copy_fmow_to_gaia7
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=log/slurm/copy_fmow_to_gaia7.%j.out
#SBATCH --error=log/slurm/copy_fmow_to_gaia7.%j.err

set -euo pipefail

srun rsync -avh --progress gaia5:/data/henicke/FMoW_LandSat/fmow_preprocessed /data/henicke/FMoW_LandSat
