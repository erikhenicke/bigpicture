#!/bin/bash
#SBATCH --partition=robolab
#SBATCH --nodelist=gaia4,gaia5
#SBATCH --job-name=reproduce_film
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8GB
#SBATCH --output=log/slurm/reproduce_film.%j.out
#SBATCH --error=log/slurm/reproduce_film.%j.err
cd /home/henicke/git/bigpicture
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/feature_fusion.yaml --run-name film 
