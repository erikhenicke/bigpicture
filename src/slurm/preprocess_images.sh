#!/usr/bin/env bash
#SBATCH --partition=robolab
#SBATCH --nodelist=gaia4,gaia5
#SBATCH --job-name=preprocess_fmow
#SBATCH --nodes=1
#SBATCH --output=log/slurm/preprocess_fmow.%j.out
#SBATCH --error=log/slurm/preprocess_fmow.%j.err


if [[ "$(hostname)" == "gaia4" ]]; then
    cd /home/henicke/git/bigpicture
    PYTHONPATH=/home/henicke/git/bigpicture/src /home/henicke/miniconda3/envs/wilds2/bin/python /home/henicke/git/bigpicture/src/dataset_creation/save_transformed_images.py --fmow-dir=/home/henicke/data --landsat-dir=/home/datasets4/FMoW_LandSat --output-dir=/data/henicke/FMoW_LandSat
fi
