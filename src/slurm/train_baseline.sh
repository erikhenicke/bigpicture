#!/bin/bash
#SBATCH --partition=robolab
#SBATCH --exclude=gaia1,gaia2,gaia3,gaia6,gaia7
#SBATCH --job-name=train_dense_net_121
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/train_dense_net_121.%j.out
#SBATCH --error=slurm/train_dense_net_121.%j.err
#SBATCH --gres=gpu:1
  
# Your commands go here
cd /home/henicke/git/bigpicture
PYTHONPATH=/home/henicke/git/bigpicture/src /home/henicke/miniconda3/envs/wilds2/bin/python /home/henicke/git/bigpicture/src/train/run_experiment.py --model_type=dense_net_121 --epochs=50 --batch_size=512 --frac=1.0 --optimizer=adam --learning_rate=1e-4 --learning_rate_decay=0.96
