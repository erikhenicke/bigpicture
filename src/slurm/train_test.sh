#!/bin/bash
#SBATCH --partition=robolab
#SBATCH --exclude=gaia1,gaia2,gaia3,gaia6,gaia7
#SBATCH --job-name=test_exp
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/test_exp.%j.out
#SBATCH --error=slurm/test_exp.%j.err
#SBATCH --gres=gpu:1

# Your commands go here
cd /home/henicke/git/bigpicture
PYTHONPATH=/home/henicke/git/bigpicture/src /home/henicke/miniconda3/envs/wilds2/bin/python /home/henicke/git/bigpicture/src/train/run_experiment.py --model_type=single-deit --epochs=100 --batch_size=10 --frac=0.01
