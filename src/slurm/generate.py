#!/usr/bin/env python3
"""Generate SLURM job scripts from a run YAML config.

Reads a YAML file under `src/train/configs/run/` describing one or more
training experiments (each specifying a model plus config overrides) and
optional SLURM resource settings, and renders one `sbatch`-ready shell
script per experiment into `src/slurm/<run_yaml_stem>/`. Each generated
script invokes `src/train/run_experiment.py` (the Hydra training
entrypoint) with `model=<model>` plus the merged Hydra override
`key=value` pairs.

Functions:
    `build_override_args`: Merge global and per-experiment config overrides
        into a Hydra `key=value ...` argument string.
    `generate`: Parse a run YAML and write one SLURM script per experiment.
    `main`: CLI entrypoint; picks a run YAML (from argv or an interactive
        prompt) and calls `generate`.

Usage:
    uv run python src/slurm/generate.py [run_yaml]
"""

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent.parent.parent
RUN_CONFIG_DIR = REPO_ROOT / "src" / "train" / "configs" / "run"
SLURM_DIR = Path(__file__).parent

NODELIST = "gaia4,gaia7"
GPUS = 1

SCRIPT_TEMPLATE = """\
#!/bin/bash
#SBATCH --partition={partition}
#SBATCH --nodelist={nodelist}
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={gpus}
#SBATCH --gres=gpu:{gpus}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem-per-cpu={mem_per_cpu}
#SBATCH --output=log/slurm/{job_name}.%j.out
#SBATCH --error=log/slurm/{job_name}.%j.err
cd {repo_root}
srun uv run --env-file .env src/train/run_experiment.py {args}
"""


def build_override_args(global_overrides: dict, run_overrides: dict, run_name: str) -> str:
    """Merge global and per-experiment overrides into a Hydra override string.

    Args:
        global_overrides (dict): Config overrides shared by all experiments
            in the run YAML (from the YAML's `global_overrides` key).
        run_overrides (dict): Per-experiment overrides (from the
            experiment's `overrides` key); take precedence over
            `global_overrides` and the injected `run_name`. May be falsy
            (e.g. `None`/`{}`), in which case only `global_overrides` and
            `run_name` apply.
        run_name (str): Value injected as the `run_name` override (the
            generated job name, e.g. `train_<experiment_key>`).

    Returns:
        str: Space-separated `key=value` Hydra override arguments, e.g.
        `"data.batch_size=32 run_name=train_foo"`.
    """
    merged = {**global_overrides, "run_name": run_name, **(run_overrides or {})}
    return " ".join(f"{k}={v}" for k, v in merged.items())


def generate(run_yaml: Path) -> list[Path]:
    """Render one SLURM script per experiment defined in a run YAML.

    Reads `run_yaml`, resolves SLURM resource settings (`slurm.partition`,
    `slurm.cpus_per_task`, `slurm.mem_per_cpu`, `slurm.nodelist`, each with a
    default) and the `global_overrides`/`experiments` sections, then writes
    one executable `.sh` script per entry in `experiments` to
    `src/slurm/<run_yaml.stem>/train_<key>.sh`, filling `SCRIPT_TEMPLATE`
    with the resource settings and a `model=<model> <merged overrides>`
    argument string for `run_experiment.py`.

    Args:
        run_yaml (Path): Path to a run config YAML (see
            `src/train/configs/run/` for examples), with top-level keys
            `slurm` (optional), `global_overrides` (optional), and
            `experiments` (required, mapping an experiment key to a dict
            with `model` and optional `overrides`).

    Returns:
        list[Path]: Paths of the generated `.sh` scripts, one per
        experiment.

    Raises:
        SystemExit: If `experiments` is empty/missing, or if an experiment
            entry does not specify a `model` key.
    """
    with run_yaml.open() as f:
        cfg = yaml.safe_load(f)

    slurm = cfg.get("slurm", {})
    partition = slurm.get("partition", "robolab")
    cpus_per_task = slurm.get("cpus_per_task", 4)
    mem_per_cpu = slurm.get("mem_per_cpu", "8G")
    nodelist = slurm.get("nodelist", NODELIST)

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
            gpus=GPUS,
            cpus_per_task=cpus_per_task,
            mem_per_cpu=mem_per_cpu,
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
    """CLI entrypoint: resolve the target run YAML and generate its SLURM scripts.

    If a `run_yaml` positional argument is given, resolves it relative to
    the current working directory (if not already absolute). Otherwise
    lists the `*.yaml` files in `RUN_CONFIG_DIR` and interactively prompts
    the user to pick one by index. Delegates the actual script generation
    to `generate`.

    Raises:
        SystemExit: If no `run_yaml` argument is given and `RUN_CONFIG_DIR`
            contains no YAML files, or if the resolved `run_yaml` path does
            not exist.
    """
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
