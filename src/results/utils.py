"""Shared helpers for loading trained runs (checkpoints + hydra config) and for
resolving eval-YAML run references to their log directories and display names.

Checkpoint/config loading: ``find_best_checkpoints`` and ``load_hydra_config``
load a run's per-seed checkpoints and its Hydra config from a log directory;
``find_run_dir`` resolves an experiment key to its most recent log directory.

Metrics loading: ``load_seed_test_metrics`` reads one seed's flat test-metric
map (preferring the rerun output over the training CSV); ``load_run_metrics``
averages that across every seed of a run.

Eval-YAML run references: ``parse_run_ref`` splits a ``"config@exp_key"`` (or
bare ``"exp_key"``) reference; ``load_run_configs`` loads the referenced run
config YAMLs; ``resolve_experiments`` maps each run reference to its resolved
experiment definition (including only the overrides that differ from the eval
YAML's own config); ``load_translations`` loads the metric/model/param
display-name translations; ``format_experiment_name`` turns a resolved
experiment definition into a human-readable (optionally LaTeX) display name.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).parent.parent.parent
LOG_RUNS = REPO_ROOT / "log" / "runs"
EVAL_CONFIG_DIR = REPO_ROOT / "src" / "train" / "configs" / "eval"
RUN_CONFIG_DIR = REPO_ROOT / "src" / "train" / "configs" / "run"
TRANSLATIONS_FILE = EVAL_CONFIG_DIR / "translations.yaml"


def find_best_checkpoints(run_dir: Path) -> list[Path]:
    """Find the best checkpoint for each seed run under ``run_dir/checkpoints``.

    Prefers ``late-fusion-*.ckpt`` (run_experiment.py's ModelCheckpoint filename),
    falling back to ``last.ckpt``. Seed dirs with neither are skipped. Returned in
    sorted ``run*`` order.

    Args:
        run_dir (Path): Run's top-level log directory, containing
            ``checkpoints/run*``.

    Returns:
        list[Path]: Best checkpoint path per seed, in sorted ``run*`` order.
    """
    checkpoints = []
    ckpt_root = run_dir / "checkpoints"
    for seed_dir in sorted(ckpt_root.glob("run*")):
        best = list(seed_dir.glob("late-fusion-*.ckpt"))
        if best:
            checkpoints.append(best[0])
        else:
            last = seed_dir / "last.ckpt"
            if last.exists():
                checkpoints.append(last)
    return checkpoints


def load_hydra_config(run_dir: Path):
    """Load ``.hydra/config.yaml`` from a run directory, backfilling trainer fields
    that ``make_model`` reads but older runs didn't persist.

    Args:
        run_dir (Path): Run's top-level log directory.

    Returns:
        omegaconf.DictConfig: The loaded Hydra config, with any missing
            ``trainer.alternating_freeze``, ``trainer.alternating_freeze_period``,
            and ``trainer.branch_ablation`` fields backfilled with their defaults.

    Raises:
        FileNotFoundError: If ``run_dir/.hydra/config.yaml`` does not exist.
    """
    config_path = Path(run_dir) / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No .hydra/config.yaml found in {run_dir}")
    cfg = OmegaConf.load(config_path)
    trainer_defaults = {
        "alternating_freeze": False,
        "alternating_freeze_period": 1,
        "branch_ablation": False,
    }
    for key, default in trainer_defaults.items():
        if key not in cfg.trainer:
            cfg.trainer[key] = default
    return cfg


def find_run_dir(exp_key: str) -> Path | None:
    """Return the most recent log directory for the given experiment key.

    Searches every ``LOG_RUNS/<date>/`` directory for entries named
    ``train_<exp_key>-*`` and returns the one with the lexicographically latest
    (date dir, run dir) pair.

    Args:
        exp_key (str): Experiment key, matched against directories named
            ``train_<exp_key>-*``.

    Returns:
        Path | None: Path to the most recent matching run directory, or None if
            no run directory matches.
    """
    job_name = f"train_{exp_key}"
    prefix = job_name + "-"

    candidates: list[tuple[str, str, Path]] = []
    for date_dir in LOG_RUNS.iterdir():
        if not date_dir.is_dir():
            continue
        for run_dir in date_dir.iterdir():
            if run_dir.name.startswith(prefix):
                candidates.append((date_dir.name, run_dir.name, run_dir))

    if not candidates:
        return None
    # Take the most recent by date, then by run name
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2]


def parse_run_ref(ref: str, default_config: str) -> tuple[str, str]:
    """Split a run reference into its run-config name and experiment key.

    Args:
        ref (str): Run reference, either ``"config_name@exp_key"`` or a bare
            ``"exp_key"``.
        default_config (str): Config name to use when ``ref`` has no ``@`` prefix.

    Returns:
        tuple[str, str]: ``(config_name, exp_key)``.
    """
    if "@" in ref:
        config_name, exp_key = ref.split("@", 1)
        return config_name, exp_key
    return default_config, ref


def load_run_configs(config_names: set[str]) -> dict[str, dict]:
    """Load and parse the run config YAML for each given config name.

    Args:
        config_names (set[str]): Run-config names (YAML stems under
            ``RUN_CONFIG_DIR``) to load.

    Returns:
        dict[str, dict]: Config name -> parsed YAML dict, for names whose file
            exists; missing files are skipped with a warning printed to stderr.
    """
    configs: dict[str, dict] = {}
    for name in config_names:
        path = RUN_CONFIG_DIR / f"{name}.yaml"
        if not path.exists():
            print(f"Warning: run config not found: {path}", file=sys.stderr)
            continue
        with path.open() as f:
            configs[name] = yaml.safe_load(f)
    return configs


def resolve_experiments(
    groups: list[dict],
    run_configs: dict[str, dict],
    default_config: str,
) -> dict[str, dict]:
    """Resolve every run reference used across a set of eval-YAML groups to its
    experiment definition.

    For each reference, merges its run config's ``global_overrides`` (restricted
    to the keys that differ from ``default_config``'s own ``global_overrides``)
    with the experiment's own ``overrides``, so the resulting display overrides
    show only what's non-default relative to the eval YAML's own config.

    Args:
        groups (list[dict]): Eval-YAML groups, each with a ``"runs"`` list of run
            references.
        run_configs (dict[str, dict]): Config name -> parsed run config YAML (see
            ``load_run_configs``).
        default_config (str): Run-config name used to determine which
            ``global_overrides`` count as "different from default".

    Returns:
        dict[str, dict]: Run reference -> resolved experiment definition (the
            experiment's own fields plus a merged ``"overrides"`` dict, or None
            if there are none). References whose config or experiment key can't
            be found are omitted (with a warning printed to stderr).
    """
    default_global = run_configs.get(default_config, {}).get("global_overrides", {})
    resolved: dict[str, dict] = {}
    for group in groups:
        for ref in group["runs"]:
            if ref in resolved:
                continue
            config_name, exp_key = parse_run_ref(ref, default_config)
            cfg = run_configs.get(config_name)
            if cfg is None:
                print(f"Warning: run config '{config_name}' not found (referenced by '{ref}')", file=sys.stderr)
                continue
            exp_def = cfg.get("experiments", {}).get(exp_key)
            if exp_def is None:
                print(f"Warning: experiment '{exp_key}' not found in '{config_name}'", file=sys.stderr)
                continue
            source_global = cfg.get("global_overrides", {})
            exp_overrides = exp_def.get("overrides") or {}
            global_diff = {k: v for k, v in source_global.items() if default_global.get(k) != v}
            display_overrides = {**global_diff, **exp_overrides}
            resolved[ref] = {**exp_def, "overrides": display_overrides or None}
    return resolved


def load_seed_test_metrics(seed_dir: Path) -> dict[str, float] | None:
    """Load one seed's flat test metric map, preferring ``metrics_rerun.csv``.

    ``metrics_rerun.csv`` is the long-format (``metric,value``) rerun output and
    the only source of the per-class / top-5 keys. Falls back to the wide training
    ``metrics.csv`` (its final non-empty ``test/`` row). Returns ``None`` if neither
    file yields test metrics.

    Args:
        seed_dir (Path): Seed's run directory (``run{i}``), containing
            ``version_0/``.

    Returns:
        dict[str, float] | None: Metric name -> value map, or None if neither
            ``version_0/metrics_rerun.csv`` nor ``version_0/metrics.csv`` yields
            test metrics.
    """
    rerun = seed_dir / "version_0" / "metrics_rerun.csv"
    if rerun.exists():
        df = pd.read_csv(rerun)
        return {str(m): float(v) for m, v in zip(df["metric"], df["value"]) if pd.notna(v)}

    plain = seed_dir / "version_0" / "metrics.csv"
    if plain.exists():
        df = pd.read_csv(plain)
        test_cols = [c for c in df.columns if c.startswith("test/")]
        if not test_cols:
            return None
        test_rows = df.dropna(subset=[test_cols[0]])
        if test_rows.empty:
            return None
        row = test_rows.iloc[-1]
        return {c: float(row[c]) for c in test_cols if pd.notna(row[c])}

    return None


def load_run_metrics(run_dir: Path | None) -> dict[str, float]:
    """Mean of each test metric across all seeds of a run (empty if none found).

    Each seed is loaded via :func:`load_seed_test_metrics`, so per-seed values come
    from ``metrics_rerun.csv`` when present, otherwise ``metrics.csv``.

    Args:
        run_dir (Path | None): Run's top-level log directory, or None.

    Returns:
        dict[str, float]: Metric name -> mean value across seeds that have it;
            empty dict if ``run_dir`` is None or no seed has any test metrics.
    """
    if run_dir is None:
        return {}
    pooled: dict[str, list[float]] = {}
    for seed_dir in sorted(run_dir.glob("run*")):
        seed_metrics = load_seed_test_metrics(seed_dir)
        if not seed_metrics:
            continue
        for k, v in seed_metrics.items():
            pooled.setdefault(k, []).append(v)
    return {k: sum(vs) / len(vs) for k, vs in pooled.items()}


def load_translations() -> dict:
    """Load the metric/model/param display-name translations YAML.

    Returns:
        dict: Parsed ``translations.yaml`` with (at least) ``"models"`` and
            ``"params"`` keys, or ``{"models": {}, "params": {}}`` if the file
            doesn't exist.
    """
    translations: dict = {"models": {}, "params": {}}
    if TRANSLATIONS_FILE.exists():
        with TRANSLATIONS_FILE.open() as f:
            translations = yaml.safe_load(f)
    return translations


def format_experiment_name(
    run_ref: str,
    run_experiments: dict,
    translations: dict,
    latex: bool = False,
    model_overrides: dict | None = None,
    param_overrides: dict | None = None,
) -> str:
    """Format experiment name using translations + optional group-level overrides.

    Args:
        run_ref (str): Run reference (e.g., ``"config@exp_key"`` or just ``"exp_key"``).
        run_experiments (dict): Run reference -> resolved experiment definition
            (see ``resolve_experiments``).
        translations (dict): Dict with ``"models"`` and ``"params"`` sections for
            global display-name translations (see ``load_translations``).
        latex (bool): Use LaTeX display strings where available, instead of plain text.
        model_overrides (dict | None): Group-level model display name overrides
            (``{model_key: display_name}``), checked before the global translations.
        param_overrides (dict | None): Group-level param display name overrides
            (``{param_key: display_name_or_dict}``); a non-dict value
            unconditionally replaces (or, if falsy, hides) the param's display, a
            dict value is merged into the global translation for that param.

    Returns:
        str: Comma-joined display name, e.g. ``"FiLM, λ=0.2"``. Falls back to a
            title-cased version of the experiment key if ``run_ref`` is not in
            ``run_experiments``.
    """
    exp_def = run_experiments.get(run_ref)
    exp_key = run_ref.split("@", 1)[1] if "@" in run_ref else run_ref
    if exp_def is None:
        return exp_key.replace("_", " ").title()

    symbol_key = "latex" if latex else "plain"
    model_key = exp_def.get("model", exp_key)

    # Check model overrides first, then global translations
    if model_overrides and model_key in model_overrides:
        base_name = model_overrides[model_key]
    else:
        base_name = translations.get("models", {}).get(model_key, model_key.replace("_", " ").title())

    overrides: dict = exp_def.get("overrides") or {}
    parts = [base_name]
    for param_key, value in overrides.items():
        param_trans = translations.get("params", {}).get(param_key)
        has_local_override = param_overrides is not None and param_key in param_overrides
        local_override = param_overrides[param_key] if has_local_override else None

        # A non-dict local override (e.g. null, or a fixed string) unconditionally
        # replaces the param's display, ignoring its value. A falsy override hides it.
        if has_local_override and not isinstance(local_override, dict):
            if local_override:
                parts.append(local_override)
            continue

        # A dict local override merges into the global translation (taking precedence),
        # so it can override e.g. just one entry of "values" without losing the rest.
        effective_trans = dict(param_trans) if param_trans else {}
        if has_local_override:
            merged_values = {**effective_trans.get("values", {}), **local_override.get("values", {})}
            effective_trans.update(local_override)
            if merged_values:
                effective_trans["values"] = merged_values

        if effective_trans.get("hidden", False):
            continue
        label = effective_trans.get(symbol_key, effective_trans.get("plain", param_key.split(".")[-1]))
        values_map = effective_trans.get("values", {})
        value_trans = values_map.get(value)
        if value_trans is None:
            value_trans = values_map.get(str(value))
        if value_trans is None and isinstance(value, bool):
            value_trans = values_map.get(str(value).lower())
        if value_trans:
            if value_trans.get("hidden", False):
                continue
            parts.append(value_trans.get(symbol_key, value_trans.get("plain", value_trans.get("latex"))))
        elif isinstance(value, bool):
            if not value:
                parts.append(f"no {label}")
            else:
                parts.append(label)
        elif effective_trans.get("no_value", False):
            parts.append(label)
        else:
            val_str = f"{value:g}" if isinstance(value, float) else str(value)
            parts.append(f"{label}$={val_str}$" if latex else f"{label}={val_str}")

    return ", ".join(parts)