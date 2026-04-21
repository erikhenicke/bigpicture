#!/usr/bin/env bash
#SBATCH --partition=robolab
#SBATCH --nodelist=gaia5
#SBATCH --job-name=copy_fmow_to_gaia5
#SBATCH --nodes=1
#SBATCH --output=log/slurm/copy_fmow_to_gaia5.%j.out
#SBATCH --error=log/slurm/copy_fmow_to_gaia5.%j.err

set -euo pipefail

rsync -avh --progress /home/datasets4/FMoW_LandSat/fmow_preprocessed_norm /data/henicke/FMoW_LandSat/

