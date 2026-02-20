#!/bin/bash
#SBATCH --partition=robolab
#SBATCH --exclude=gaia1,gaia2,gaia3,gaia5,gaia6,gaia7
#SBATCH --job-name=train_sm_deit
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/train_sm_deit.%j.out
#SBATCH --error=slurm/train_sm_deit.%j.err
#SBATCH --gres=gpu:1
  
# Your commands go here
cd /home/henicke/git/bigpicture
PYTHONPATH=/home/henicke/git/bigpicture/src /home/henicke/miniconda3/envs/wilds2/bin/python /home/henicke/git/bigpicture/src/train/run_experiment.py --model_type=multi --epochs=50 --batch_size=384 --frac=1.0 --optimizer=adamw --learning_rate=5e-5 --weight_decay=0.01
PYTHONPATH=/home/henicke/git/bigpicture/src /home/henicke/miniconda3/envs/wilds2/bin/python /home/henicke/git/bigpicture/src/train/run_experiment.py --model_type=single --epochs=50 --batch_size=384 --frac=1.0 --optimizer=adamw --learning_rate=5e-5 --weight_decay=0.01
