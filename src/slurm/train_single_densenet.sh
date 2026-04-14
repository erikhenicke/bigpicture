#!/bin/bash
#SBATCH --partition=robolab
#SBATCH --exclude=gaia1,gaia2,gaia3,gaia6,gaia7
#SBATCH --job-name=train_single_densenet
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=log/slurm/train_single_densenet.%j.out
#SBATCH --error=log/slurm/train_single_densenet.%j.err
#SBATCH --gres=gpu:1
  
# Your commands go here
cd /home/henicke/git/bigpicture

PYTHONPATH=/home/henicke/git/bigpicture/src /home/henicke/miniconda3/envs/wilds2/bin/python /home/henicke/git/bigpicture/src/train/run_experiment.py --model_type=single-densenet --data_augmentation --image_net=hr --epochs=50 --batch_size=64 --frac=1 --optimizer=adam --learning_rate=1e-4 --learning_rate_scheduler=step --learning_rate_decay=0.96
# PYTHONPATH=/home/henicke/git/bigpicture/src /home/henicke/miniconda3/envs/wilds2/bin/python /home/henicke/git/bigpicture/src/train/run_experiment.py --model_type=single-densenet --image_net=none --epochs=50 --batch_size=64 --frac=1 --optimizer=adam --learning_rate=1e-4 --learning_rate_scheduler=step --learning_rate_decay=0.96
