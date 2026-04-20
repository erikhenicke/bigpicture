#!/usr/bin/env python3
"""Generate SLURM job scripts from a run YAML config."""

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent.parent.parent
RUN_CONFIG_DIR = REPO_ROOT / "src" / "train" / "configs" / "run"
SLURM_DIR = Path(__file__).parent

NODES: dict[str, dict[str, int]] = {
    "gaia4": {"gpus": 1, "throughput": 1},
    "gaia5": {"gpus": 2, "throughput": 2},
}

SCRIPT_TEMPLATE = """\
#!/bin/bash
#SBATCH --partition={partition}
#SBATCH --nodelist={nodelist}
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --gres=gpu:{gpus}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --output=log/slurm/{job_name}.%j.out
#SBATCH --error=log/slurm/{job_name}.%j.err
cd {repo_root}
uv run --env-file .env src/train/run_experiment.py trainer.devices=$SLURM_GPUS_ON_NODE {args}
"""


def build_override_args(global_overrides: dict, run_overrides: dict, run_name: str) -> str:
    merged = {**global_overrides, "run_name": run_name, **(run_overrides or {})}
    return " ".join(f"{k}={v}" for k, v in merged.items())


def schedule_experiments(experiments: dict) -> dict[str, str]:
    """LPT heuristic on related machines: sort jobs by load descending, assign each
    to the node where it would finish earliest (current_time + load / throughput)."""
    sorted_keys = sorted(
        experiments.keys(),
        key=lambda k: experiments[k].get("load", 1),
        reverse=True,
    )
    node_time: dict[str, float] = {name: 0.0 for name in NODES}
    assignment: dict[str, str] = {}
    for key in sorted_keys:
        load = experiments[key].get("load", 1)
        best_node = min(
            NODES,
            key=lambda n: node_time[n] + load / NODES[n]["throughput"],
        )
        node_time[best_node] += load / NODES[best_node]["throughput"]
        assignment[key] = best_node
    return assignment


def print_schedule(experiments: dict, assignment: dict[str, str]) -> None:
    by_node: dict[str, list[str]] = {name: [] for name in NODES}
    for key, node in assignment.items():
        by_node[node].append(key)
    print("\nLoad balancing:")
    for node, keys in by_node.items():
        throughput = NODES[node]["throughput"]
        total_load = sum(experiments[k].get("load", 1) for k in keys)
        est_time = total_load / throughput if throughput else 0.0
        print(f"  {node} (throughput={throughput}, total_load={total_load}, est_time={est_time:.1f}):")
        for k in keys:
            print(f"    - {k} (load={experiments[k].get('load', 1)})")


def generate(run_yaml: Path) -> list[Path]:
    with run_yaml.open() as f:
        cfg = yaml.safe_load(f)

    slurm = cfg.get("slurm", {})
    partition = slurm.get("partition", "robolab")
    cpus_per_gpu = slurm.get("cpus_per_gpu", 4)

    global_overrides = cfg.get("global_overrides", {})
    experiments = cfg.get("experiments", {})
    if not experiments:
        print(f"No experiments defined in {run_yaml}", file=sys.stderr)
        sys.exit(1)

    assignment = schedule_experiments(experiments)

    out_dir = SLURM_DIR / run_yaml.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    for key, exp in experiments.items():
        model = exp.get("model")
        if not model:
            print(f"Experiment '{key}' must specify a 'model' key.", file=sys.stderr)
            sys.exit(1)

        node = assignment[key]
        node_info = NODES[node]

        job_name = f"train_{key}"
        run_overrides = exp.get("overrides", {})
        override_args = build_override_args(global_overrides, run_overrides, job_name)
        args = f"model={model} {override_args}"

        script = SCRIPT_TEMPLATE.format(
            partition=partition,
            nodelist=node,
            job_name=job_name,
            gpus=node_info["gpus"],
            cpus_per_task=cpus_per_gpu * node_info["gpus"],
            repo_root=REPO_ROOT,
            args=args,
        )

        out_path = out_dir / f"{job_name}.sh"
        out_path.write_text(script)
        out_path.chmod(0o755)
        generated.append(out_path)
        print(f"  wrote {out_path.relative_to(REPO_ROOT)} -> {node}")

    print_schedule(experiments, assignment)

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
