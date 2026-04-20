#!/usr/bin/env python3
"""Generate SLURM job scripts from a run YAML config."""

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent.parent.parent
RUN_CONFIG_DIR = REPO_ROOT / "src" / "train" / "configs" / "run"
SLURM_DIR = Path(__file__).parent

SCRIPT_TEMPLATE = """\
#!/bin/bash
#SBATCH --partition={partition}
#SBATCH --nodelist={nodelist}
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu={cpus_per_gpu}
#SBATCH --output=log/slurm/{job_name}.%j.out
#SBATCH --error=log/slurm/{job_name}.%j.err
#SBATCH --gres=gpu:{gpus}
cd {repo_root}
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
uv run --env-file .env src/train/run_experiment.py trainer.devices=$NUM_GPUS {args}
"""


def build_override_args(global_overrides: dict, run_overrides: dict, run_name: str) -> str:
    merged = {**global_overrides, "run_name": run_name, **(run_overrides or {})}
    return " ".join(f"{k}={v}" for k, v in merged.items())


def generate(run_yaml: Path) -> list[Path]:
    with run_yaml.open() as f:
        cfg = yaml.safe_load(f)

    slurm = cfg.get("slurm", {})
    partition = slurm.get("partition", "robolab")
    nodelist = slurm.get("nodelist", "gaia4,gaia5")
    cpus_per_gpu = slurm.get("cpus_per_gpu", 4)
    gpus = slurm.get("gpus", 1)

    global_overrides = cfg.get("global_overrides", {})
    experiments = cfg.get("experiments", {})
    if not experiments:
        print(f"No experiments defined in {run_yaml}", file=sys.stderr)
        sys.exit(1)

    out_dir = SLURM_DIR / run_yaml.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    for key, exp in experiments.items():
        model = exp.get("model")
        if not model:
            print(f"Experiment '{key}' must specify a 'model' key.", file=sys.stderr)
            sys.exit(1)

        job_name = f"train_{key}"
        run_overrides = exp.get("overrides", {})
        override_args = build_override_args(global_overrides, run_overrides, job_name)
        args = f"model={model} {override_args}"

        script = SCRIPT_TEMPLATE.format(
            partition=partition,
            nodelist=nodelist,
            job_name=job_name,
            cpus_per_gpu=cpus_per_gpu,
            gpus=gpus,
            repo_root=REPO_ROOT,
            args=args,
        )

        out_path = out_dir / f"{job_name}.sh"
        out_path.write_text(script)
        out_path.chmod(0o755)
        generated.append(out_path)
        print(f"  wrote {out_path.relative_to(REPO_ROOT)}")

    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate SLURM scripts from run YAML")
    parser.add_argument(
        "run_yaml",
        nargs="?",
        help="Path to run YAML (default: prompt user to pick from src/train/configs/run/)",
    )
    args = parser.parse_args()

    if args.run_yaml:
        run_yaml = Path(args.run_yaml)
        if not run_yaml.is_absolute():
            run_yaml = Path.cwd() / run_yaml
    else:
        yamls = sorted(RUN_CONFIG_DIR.glob("*.yaml"))
        if not yamls:
            print(f"No YAML files found in {RUN_CONFIG_DIR}", file=sys.stderr)
            sys.exit(1)
        print("Available run configs:")
        for i, p in enumerate(yamls):
            print(f"  [{i}] {p.name}")
        idx = int(input("Select config index: "))
        run_yaml = yamls[idx]

    if not run_yaml.exists():
        print(f"File not found: {run_yaml}", file=sys.stderr)
        sys.exit(1)

    print(f"Generating scripts from {run_yaml.name}:")
    generate(run_yaml)


if __name__ == "__main__":
    main()
