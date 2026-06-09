#!/usr/bin/env bash
#SBATCH --partition=robolab
#SBATCH --nodelist=gaia5
#SBATCH --job-name=preprocess_fmow_fullres
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=log/slurm/preprocess_fmow_fullres.%j.out
#SBATCH --error=log/slurm/preprocess_fmow_fullres.%j.err

set -euo pipefail

# Native-resolution (498x498), normalized, fp16 Landsat tensors for the
# spatial-extent ablation. Lands in <OUTDIR>/landsat/image_{idx}.pt
# (the script appends the landsat modality subdir).
OUTDIR="/data/henicke/FMoW_LandSat_FullRes"

# Must match the normalization the consuming training runs expect.
IMAGE_NORM="fmow-statistics"

mkdir -p "$OUTDIR"
cd /home/henicke/git/bigpicture

# Not idempotent-destructive: the python script skips already-written files, so
# a re-submitted job resumes. Clear "$OUTDIR/landsat" by hand if you
# change IMAGE_NORM or the source data.
srun uv run --env-file .env src/dataset_creation/save_fullres_landsat.py \
    --fmow-dir=/home/henicke/data \
    --landsat-dir=/home/datasets4/FMoW_LandSat \
    --output-dir="$OUTDIR" \
    --image-norm="$IMAGE_NORM"
