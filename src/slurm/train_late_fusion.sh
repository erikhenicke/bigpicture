#!/bin/bash
#SBATCH --partition=robolab
#SBATCH --nodelist=gaia4,gaia5
#SBATCH --job-name=train_late_fusion
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=log/slurm/train_late_fusion.%j.out
#SBATCH --error=log/slurm/train_late_fusion.%j.err
#SBATCH --gres=gpu:1

cd /home/henicke/git/bigpicture
uv run --env-file .env src/train/run_experiment.py experiment=train_multi_deit_d3g
