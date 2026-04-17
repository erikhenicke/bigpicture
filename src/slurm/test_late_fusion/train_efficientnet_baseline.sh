#!/bin/bash
#SBATCH --partition=robolab
#SBATCH --nodelist=gaia4,gaia5
#SBATCH --job-name=train_efficientnet_baseline
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=log/slurm/train_efficientnet_baseline.%j.out
#SBATCH --error=log/slurm/train_efficientnet_baseline.%j.err
#SBATCH --gres=gpu:1
cd /home/erik/git/bigpicture
uv run --env-file .env src/train/run_experiment_lightning_hydra.py experiment=train_efficientnet_baseline data.frac=0.01 trainer.max_epochs=5
