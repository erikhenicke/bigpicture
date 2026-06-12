#!/bin/bash
#SBATCH --partition=robolab
#SBATCH --nodelist=gaia4,gaia7
#SBATCH --job-name=reproduce
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8GB
#SBATCH --output=log/slurm/reproduce.%j.out
#SBATCH --error=log/slurm/reproduce.%j.err
cd /home/henicke/git/bigpicture

# ---------------- baselines.yaml (13 runs with 3 seeds) ----------------
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/baselines.yaml --run-name densenet_baseline
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/baselines.yaml --run-name efficientformer_baseline
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/baselines.yaml --run-name efficientnet_baseline
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/baselines.yaml --run-name deit_baseline
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/baselines.yaml --run-name satclip_image_enc_baseline
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/baselines.yaml --run-name satclip_le_enc_baseline
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/baselines.yaml --run-name satclip_le_enc_baseline_no_domain
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/baselines.yaml --run-name dinov3_baseline
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/baselines.yaml --run-name dinov3_lr_baseline
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/baselines.yaml --run-name dinov3_lr_baseline_no_domain
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/baselines.yaml --run-name densenet_lr_baseline
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/baselines.yaml --run-name densenet_lr_baseline_no_domain
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/baselines.yaml --run-name densenet_baseline_constnorm

# ---------------- multsim.yaml (35 runs with 3 seeds) ----------------
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name concat
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name film
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name geoprior
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name d3g
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name multsim
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name multsim_detach
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name concat_detach
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name d3g_detach_lr
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name d3g_detach_hr
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name d3g_detach_both
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name d3g_detach_both_constnorm
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name film_detach
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name film_no_domain
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name concat_no_domain
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name geoprior_no_domain
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name d3g_no_domain
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name multsim_no_relu
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name film_relu
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name multsim_ls_0_1
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name multsim_constnorm
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name concat_constnorm
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name multsim_om_gauss
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name multsim_om_gauss_constnorm
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name multsim_pe
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name multsim_pe_freq
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name film_om_bin
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name film_om_gauss
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name film_pe
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name film_pe_freq
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name film_om_gauss_pe_freq
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name film_om_bin_pe_freq
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name film_om_bin_pe
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name film_om_gauss_pe
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name multsim_alternate_branches
srun uv run --env-file .env src/results/eval_reproduce.py --config src/train/configs/run/multsim.yaml --run-name multsim_alternate_om_gauss
