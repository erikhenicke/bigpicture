#!/bin/bash
#SBATCH --partition=robolab
#SBATCH --nodelist=gaia4,gaia5
#SBATCH --job-name=reproduce_dinov3_baseline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8GB
#SBATCH --output=log/slurm/reproduce_dinov3_baseline.%j.out
#SBATCH --error=log/slurm/reproduce_dinov3_baseline.%j.err
cd /home/henicke/git/bigpicture
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/baselines.yaml --run-name dinov3_baseline
