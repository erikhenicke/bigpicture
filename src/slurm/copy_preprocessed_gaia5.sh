#!/usr/bin/env bash
#SBATCH --partition=robolab
#SBATCH --nodelist=gaia5
#SBATCH --job-name=copy_fmow_to_gaia5
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=log/slurm/copy_fmow_to_gaia5.%j.out
#SBATCH --error=log/slurm/copy_fmow_to_gaia5.%j.err

set -euo pipefail

srun rsync -avh --progress gaia7:/data/henicke/FMoW_LandSat_Norm_/fmow_preprocessed/landsat /data/henicke/FMoW_LandSat_Norm_/fmow_preprocessed/

