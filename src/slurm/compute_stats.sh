#!/bin/bash
#SBATCH --partition=robolab
#SBATCH --job-name=compute_stats
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=log/slurm/compute_stats.%j.out
#SBATCH --error=log/slurm/compute_stats.%j.err

# Your commands go here
cd /home/henicke/git/bigpicture
uv run --env-file .env src/dataset_creation/compute_stats.py --fmow-dir=/home/henicke/data --landsat-dir=/home/datasets4/FMoW_LandSat --output-json=log/stats.json

