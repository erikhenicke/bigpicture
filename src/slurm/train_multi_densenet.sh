#!/bin/bash
#SBATCH --partition=robolab
#SBATCH --exclude=gaia1,gaia2,gaia3,gaia6,gaia7
#SBATCH --job-name=train_multi_dense_net_121
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/train_multi_dense_net_121.%j.out
#SBATCH --error=slurm/train_multi_dense_net_121.%j.err
#SBATCH --gres=gpu:1
  
# Your commands go here
cd /home/henicke/git/bigpicture
PYTHONPATH=/home/henicke/git/bigpicture/src /home/henicke/miniconda3/envs/wilds2/bin/python /home/henicke/git/bigpicture/src/train/run_experiment.py --model_type=multi-dense-net-121 --epochs=100 --batch_size=384 --frac=0.01 --optimizer=adam --learning_rate=1e-4 --learning_rate_decay=0.96
PYTHONPATH=/home/henicke/git/bigpicture/src /home/henicke/miniconda3/envs/wilds2/bin/python /home/henicke/git/bigpicture/src/train/run_experiment.py --model_type=multi-dense-net-121 --epochs=100 --batch_size=384 --frac=1.0 --optimizer=adamw --learning_rate=1e-5 --weight_decay=0.01 