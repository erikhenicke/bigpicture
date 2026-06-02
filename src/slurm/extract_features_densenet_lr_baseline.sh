#!/bin/bash
#SBATCH --partition=robolab
#SBATCH --nodelist=gaia4,gaia5
#SBATCH --job-name=extract_features_densenet_lr_baseline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8GB
#SBATCH --output=log/slurm/extract_features_densenet_lr_baseline.%j.out
#SBATCH --error=log/slurm/extract_features_densenet_lr_baseline.%j.err
cd /home/henicke/git/bigpicture

# extract_features.py takes the run directory positionally; resolve the most
# recent log dir for this experiment (same selection as eval_reproduce.find_run_dir).
RUN_DIR=$(ls -d log/runs/*/train_densenet_lr_baseline-* 2>/dev/null | sort | tail -n1)
if [ -z "$RUN_DIR" ]; then
  echo "No run directory found for densenet_lr_baseline under log/runs/" >&2
  exit 1
fi

echo "Using run directory: $RUN_DIR"
start_ts=$(date -Iseconds)
start_epoch=$(date +%s)
echo "srun start: $start_ts"

if ! srun uv run --env-file .env src/results/extract_features.py "$RUN_DIR"; then
  end_ts=$(date -Iseconds)
  end_epoch=$(date +%s)
  elapsed=$((end_epoch - start_epoch))
  echo "srun end: $end_ts (failed, elapsed ${elapsed}s)" >&2
  exit 1
fi

end_ts=$(date -Iseconds)
end_epoch=$(date +%s)
elapsed=$((end_epoch - start_epoch))
echo "srun end: $end_ts (elapsed ${elapsed}s)"
