#!/usr/bin/env bash
#SBATCH --partition=robolab
#SBATCH --exclude=gaia1,gaia2,gaia3,gaia5,gaia6,gaia7
#SBATCH --job-name=preprocess_fmow
#SBATCH --nodes=1
#SBATCH --output=preprocess_fmow.%j.out
#SBATCH --error=preprocess_fmow.%j.err


if [[ "$(hostname)" == "gaia4" ]]; then
    cd /home/henicke/git/bigpicture
    PYTHONPATH=/home/henicke/git/bigpicture/src /home/henicke/miniconda3/envs/wilds2/bin/python /home/henicke/git/bigpicture/src/train/save_transformed_images.py --fmow-dir=/home/henicke/git/bigpicture/data --landsat-dir=/home/datasets4/FMoW_LandSat --output-dir=/data/henicke/FMoW_LandSat
fi
