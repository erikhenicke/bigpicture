#!/bin/bash
#SBATCH --partition=robolab
#SBATCH --nodelist=gaia4,gaia7
#SBATCH --job-name=lr_domain_probe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8GB
#SBATCH --output=log/slurm/lr_domain_probe.%j.out
#SBATCH --error=log/slurm/lr_domain_probe.%j.err
cd /home/henicke/git/bigpicture

# Linear-probe the LR domain head of the *_no_domain runs (trained with
# lr_domain_loss_coeff=0, so their logged lr-domain-acc is from an untrained
# head). Each run retrains only the head on frozen LR features and updates the
# LR-domain test metrics in that run's per-seed metrics file. All five runs
# live in feature_fusion.yaml.
srun uv run --env-file .env src/results/retrain_lr_domain.py --config src/train/configs/run/feature_fusion.yaml --run-name d3g_detach_lr_no_domain
