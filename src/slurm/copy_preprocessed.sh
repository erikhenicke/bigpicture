#!/usr/bin/env bash
#SBATCH --partition=robolab
#SBATCH --exclude=gaia1,gaia2,gaia3,gaia4,gaia6,gaia7
#SBATCH --job-name=copy_fmow
#SBATCH --nodes=1
#SBATCH --output=slurm/copy_fmow.%j.out
#SBATCH --error=slurm/copy_fmow.%j.err

set -euo pipefail

if [[ "$(hostname)" == "gaia5" ]]; then
    rsync -avh --progress gaia4:/data/henicke/FMoW_LandSat /data/henicke
fi
