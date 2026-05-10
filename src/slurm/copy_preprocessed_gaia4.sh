#!/usr/bin/env bash
#SBATCH --partition=robolab
#SBATCH --nodelist=gaia4
#SBATCH --job-name=copy_fmow_to_gaia4
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=log/slurm/copy_fmow_to_gaia4.%j.out
#SBATCH --error=log/slurm/copy_fmow_to_gaia4.%j.err

set -euo pipefail

mkdir -p /data/henicke/FMoW_LandSat_/fmow_preprocessed/
ln -s /data/henicke/FMoW_LandSat_Norm/fmow_preprocessed/fmow_rgb /data/henicke/FMoW_LandSat_/fmow_preprocessed/fmow_rgb
srun rsync -avh --progress /home/datasets4/FMoW_LandSat/fmow_preprocessed/landsat /data/henicke/FMoW_LandSat_/fmow_preprocessed/

