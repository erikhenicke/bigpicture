#!/bin/bash
#SBATCH --partition=robolab
#SBATCH --job-name=test_training
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=test_training.%j.out
#SBATCH --error=test_training.%j.err
#SBATCH --gres=gpu:1
  
# Your commands go here
cd /home/henicke/git/bigpicture
PYTHONPATH=/home/henicke/git/bigpicture/src /home/henicke/miniconda3/envs/wilds2/bin/python /home/henicke/git/bigpicture/src/train/train_first.py --model_type=multi --epochs=1 --batch_size=10 --frac=0.01
