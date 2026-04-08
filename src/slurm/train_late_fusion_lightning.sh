#!/bin/bash
#SBATCH --partition=robolab
#SBATCH --nodelist=gaia4,gaia5
#SBATCH --job-name=train_late_fusion_lightning
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/train_late_fusion_lightning.%j.out
#SBATCH --error=slurm/train_late_fusion_lightning.%j.err
#SBATCH --gres=gpu:1

cd /home/henicke/git/bigpicture
uv run --env-file .env src/train/run_experiment_lightning.py
