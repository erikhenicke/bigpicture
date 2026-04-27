#!/usr/bin/env bash
#SBATCH --partition=robolab
#SBATCH --nodelist=gaia4,gaia5,gaia7,nyx
#SBATCH --job-name=preprocess_fmow
#SBATCH --nodes=1
#SBATCH --output=log/slurm/preprocess_fmow.%j.out
#SBATCH --error=log/slurm/preprocess_fmow.%j.err


HOSTNAME="$(hostname)"

if [[ "$HOSTNAME" == "gaia4" || "$HOSTNAME" == "gaia5" || "$HOSTNAME" == "gaia7" ]]; then
    OUTDIR="/data/henicke/FMoW_LandSat_Norm_"
elif [[ "$HOSTNAME" == "nyx" ]]; then
    OUTDIR="/home/nyx_data1/henicke/FMoW_LandSat_Norm_"
else
    exit 0
fi

mkdir -p "$OUTDIR"
cd /home/henicke/git/bigpicture
srun uv run --env-file .env src/dataset_creation/save_transformed_images.py --fmow-dir=/home/henicke/data --landsat-dir=/home/datasets4/FMoW_LandSat --output-dir="$OUTDIR"
