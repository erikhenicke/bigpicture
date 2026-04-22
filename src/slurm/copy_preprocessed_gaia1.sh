#!/usr/bin/env bash
#SBATCH --partition=robolab
#SBATCH --nodelist=gaia1
#SBATCH --job-name=copy_fmow_to_gaia1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=log/slurm/copy_fmow_to_gaia1.%j.out
#SBATCH --error=log/slurm/copy_fmow_to_gaia1.%j.err

set -euo pipefail

srun rsync -avh --progress /home/datasets4/FMoW_LandSat/fmow_preprocessed_norm /users/henicke/FMoW_LandSat_Norm/
