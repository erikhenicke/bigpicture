#!/usr/bin/env bash
#SBATCH --partition=robolab
#SBATCH --exclude=gaia1,gaia2,gaia3,gaia5,gaia6,gaia7
#SBATCH --job-name=copy_fmow
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=copy_fmow.%j.out
#SBATCH --error=copy_fmow.%j.err

set -euo pipefail

if [[ "$(hostname)" == "gaia4" ]]; then
    rsync -avh --progress /home/datasets4/FMoW_LandSat/fmow_preprocessed /data/henicke/FMoW_LandSat
fi
