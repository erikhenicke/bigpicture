"""
utils.py

Evaluation-state bookkeeping and metric computation shared by
`MultiScaleClassificationModule` (in `multi_scale_classification.py`) across
its validation and test loops.

The functions fall into three groups:
    - Domain-label-space helpers (`domain_label_names`, `build_domain_remap`)
      that translate between raw WILDS region codes and the (possibly
      Leave-Asia-Out-reduced) label space of the domain classifiers.
    - State management (`make_eval_state`, `extract_region_names`): a plain
      dict accumulator built fresh per epoch/dataloader and mutated in place
      by the `update_*` functions as batches stream through, then reduced by
      the `compute_final_*` functions at epoch end.
    - Per-batch `update_*` functions and their matching epoch-end
      `compute_final_*` functions, covering task accuracy/loss/entropy/ECE,
      per-region and per-class breakdowns, domain-classifier accuracy, and
      branch-ablation accuracy. `update_eval_metrics` / `compute_final_eval_metrics`
      are the top-level entry points that wrap the others and prefix metric
      names with the dataloader's name for logging.

None of these functions are `nn.Module`s; the state dict they operate on is a
plain `Dict[str, Any]` of running sums/counters (see `make_eval_state`), not a
tensor.
"""
from collections import defaultdict
from typing import Any, Dict, List

import torch
from torchmetrics.classification import MulticlassCalibrationError
from wilds.datasets.fmow_dataset import categories as TASK_CLASSES


REGIONS = {0: "Asia", 1: "Europe", 2: "Africa", 3: "Americas", 4: "Oceania", 5: "Other"}

# Per-region reporting allow-list: which regions get an individually logged metric.
# "Other" is intentionally excluded in every setting (kept for training, not
# reported); Asia is listed but only has samples to report on outside LAO.
DOMAIN_NAMES = ["Asia", "Europe", "Africa", "Americas", "Oceania"]

# Number of classes to consider for top-k task accuracy.
TOPK = 5


def domain_label_names(leave_asia_out: bool) -> List[str]:
    """Label space of the domain classifier: every region present during training.

    Drives the head's output classes and the target remap. The full setting uses
    all six WILDS regions; Leave-Asia-Out drops only Asia at index 0
    (absent from training). "Other" is kept in both cases as it
    is a real trained class. Which regions are surfaced in logs is a separate
    decision: :data:`DOMAIN_NAMES` drops "Other" from per-region reporting in both
    settings.

    Args:
        leave_asia_out (bool): If True, drop "Asia" from the label space (the
            Leave-Asia-Out training setting); otherwise use all six regions.

    Returns:
        List[str]: Domain-classifier class names, in class-index order (5 names
            under Leave-Asia-Out, 6 otherwise).
    """
    regions = {k: v for k, v in REGIONS.items() if k != 0} if leave_asia_out else REGIONS
    return list(regions.values())


def build_domain_remap(domain_names: List[str]) -> torch.Tensor:
    """Map raw WILDS region codes to contiguous domain-class indices.

    Region codes whose name is not in ``domain_names`` map to ``-1``. Under
    Leave-Asia-Out only Asia maps to ``-1``; it never appears in training (so the
    domain loss never sees it) and is masked out of the domain metrics at eval,
    where domain prediction is meaningless. The full-region case yields the
    identity map.

    Args:
        domain_names (List[str]): Domain-classifier label space, as returned by
            `domain_label_names`; each name's position is its class index.

    Returns:
        torch.Tensor: Shape `(len(REGIONS),)` = `(6,)`, dtype `torch.long`. Entry
            `remap[code]` is the domain-class index for raw region `code`, or -1
            if that region's name is not in `domain_names`. Indexing this tensor
            with a batch of raw region codes (e.g. `remap[regions]`) yields the
            remapped domain targets.
    """
    name_to_class = {name: idx for idx, name in enumerate(domain_names)}
    remap = torch.full((len(REGIONS),), -1, dtype=torch.long)
    for code, name in REGIONS.items():
        if name in name_to_class:
            remap[code] = name_to_class[name]
    return remap

def make_eval_state() -> Dict[str, Any]:
    """Create a fresh accumulator dict for one epoch of one dataloader.

    Returns:
        Dict[str, Any]: Mutable state dict with running counters/sums (`total`,
            `batch_count`, per-region/per-class `defaultdict` counters,
            domain-prediction/target lists, etc.), all zero-initialized.
            Mutated in place by the `update_*` functions and consumed by the
            `compute_final_*` functions at epoch end.
    """
    return {
        "total": 0,
        "batch_count": 0,
        "region_total": defaultdict(int),
        "task_loss_sum": 0.0,
        "task_correct": 0,
        "lr_ablated_task_correct": 0,
        "hr_ablated_task_correct": 0,
        "lr_ablated_task_region_correct": defaultdict(int),
        "hr_ablated_task_region_correct": defaultdict(int),
        "task_entropy_correct": 0.0,
        "task_entropy_incorrect": 0.0,
        "task_top5_correct": 0,
        "task_top5_region_correct": defaultdict(int),
        "task_class_correct": defaultdict(int),
        "task_class_total": defaultdict(int),
        "task_region_class_correct": defaultdict(int),
        "task_region_class_total": defaultdict(int),
        "task_region_correct": defaultdict(int),
        "task_region_loss_sum": defaultdict(float),
        "lr_domain_preds": [],
        "lr_domain_targets": [],
        "hr_domain_preds": [],
        "hr_domain_targets": [],
    }

def extract_region_names(dataloaders: List[torch.utils.data.DataLoader]) -> Dict[int, List[str]]:
    """Look up each dataloader's region-name list from its underlying dataset.

    Reads `_metadata_map["region"]` off each dataloader's dataset (see WILDS
    `fmow_dataset.py`), for turning raw region codes into human-readable names
    in metric keys.

    Args:
        dataloaders (List[torch.utils.data.DataLoader]): One per eval split.

    Returns:
        Dict[int, List[str]]: Dataloader index -> region names indexed by raw
            region code, or `[]` if the dataset exposes no region metadata map.
    """
    region_names_by_loader: Dict[int, List[str]] = {}
    for idx, loader in enumerate(dataloaders):
        dataset = getattr(loader, "dataset", None)
        base_dataset = getattr(dataset, "dataset", dataset)
        metadata_map = getattr(base_dataset, "_metadata_map", None)
        if metadata_map and "region" in metadata_map:
            region_names_by_loader[idx] = list(metadata_map["region"])
        else:
            region_names_by_loader[idx] = []
    return region_names_by_loader

def update_task_entropy_metrics(
    state: Dict[str, Any],
    probs: torch.Tensor,
    correct_mask: torch.Tensor,
) -> None:
    """Accumulate summed predictive entropy of correct vs. incorrect predictions.

    Mutates `state` in place; the running sums are averaged later by
    `compute_final_task_entropy_metrics`.

    Args:
        state (Dict[str, Any]): Evaluation state dict, as created by
            `make_eval_state`.
        probs (torch.Tensor): Shape `(batch_size, num_task_labels)`, float,
            softmax class probabilities.
        correct_mask (torch.Tensor): Shape `(batch_size,)`, bool, whether each
            prediction matches the target.
    """
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
    if correct_mask.any():
        state["task_entropy_correct"] += entropy[correct_mask].sum().item()
    if (~correct_mask).any():
        state["task_entropy_incorrect"] += entropy[~correct_mask].sum().item()

def compute_final_task_entropy_metrics(state: Dict[str, Any]) -> Dict[str, float]:
    """Compute mean predictive entropy, separately for correct and incorrect predictions.

    Args:
        state (Dict[str, Any]): Evaluation state dict accumulated over an epoch
            by `update_task_entropy_metrics`.

    Returns:
        Dict[str, float]: Keys `task-entropy-correct` / `task-entropy-incorrect`
            (`torch.nan` if the respective count is 0).
    """
    entropy_correct = state["task_entropy_correct"] / state["task_correct"] if state["task_correct"] > 0 else torch.nan
    entropy_incorrect = (
        state["task_entropy_incorrect"] / (state["total"] - state["task_correct"])
        if (state["total"] - state["task_correct"]) > 0
        else torch.nan
    )
    return {
        "task-entropy-correct": entropy_correct,
        "task-entropy-incorrect": entropy_incorrect,
    }

def update_task_region_metrics(
    state: Dict[str, Any],
    metadata: torch.Tensor,
    region_index: int,
    correct_mask: torch.Tensor,
    top5_correct_mask: torch.Tensor,
    per_sample_loss: torch.Tensor,
) -> None:
    """Accumulate per-region sample counts, correct counts, and loss for one batch.

    Mutates `state`'s per-region counters in place.

    Args:
        state (Dict[str, Any]): Evaluation state dict, as created by
            `make_eval_state`.
        metadata (torch.Tensor): Shape `(batch_size, 6)`, float32, columns
            `[region, year, y, lat, lon, img_span_km]`.
        region_index (int): Column index of the region field in `metadata`.
        correct_mask (torch.Tensor): Shape `(batch_size,)`, bool, whether each
            top-1 prediction matches the target.
        top5_correct_mask (torch.Tensor): Shape `(batch_size,)`, bool, whether
            the target is among each sample's top-5 predictions.
        per_sample_loss (torch.Tensor): Shape `(batch_size,)`, float, per-sample
            (unreduced) task loss.
    """
    region_ids = metadata[:, region_index]
    for rid in torch.unique(region_ids):
        rid_int = int(rid.item())
        mask = region_ids == rid
        state["region_total"][rid_int] += int(mask.sum().item())
        state["task_region_correct"][rid_int] += int((correct_mask & mask).sum().item())
        state["task_top5_region_correct"][rid_int] += int((top5_correct_mask & mask).sum().item())
        state["task_region_loss_sum"][rid_int] += float(per_sample_loss[mask].sum().item())


def update_task_class_metrics(
    state: Dict[str, Any],
    y: torch.Tensor,
    metadata: torch.Tensor,
    region_index: int,
    correct_mask: torch.Tensor,
) -> None:
    """Update the per-class task accuracy counters (overall and per region).

    Counters are accumulated every batch but only consumed at test time (see
    `compute_final_task_class_metrics`).

    Args:
        state (Dict[str, Any]): Evaluation state dict, as created by
            `make_eval_state`.
        y (torch.Tensor): Shape `(batch_size,)`, long, ground-truth class labels.
        metadata (torch.Tensor): Shape `(batch_size, 6)`, float32, columns
            `[region, year, y, lat, lon, img_span_km]`.
        region_index (int): Column index of the region field in `metadata`.
        correct_mask (torch.Tensor): Shape `(batch_size,)`, bool, whether each
            top-1 prediction matches the target.
    """
    region_ids = metadata[:, region_index]
    for c in torch.unique(y):
        c_int = int(c.item())
        cmask = y == c
        state["task_class_total"][c_int] += int(cmask.sum().item())
        state["task_class_correct"][c_int] += int((correct_mask & cmask).sum().item())
        for rid in torch.unique(region_ids[cmask]):
            rid_int = int(rid.item())
            rcmask = cmask & (region_ids == rid)
            state["task_region_class_total"][(rid_int, c_int)] += int(rcmask.sum().item())
            state["task_region_class_correct"][(rid_int, c_int)] += int(
                (correct_mask & rcmask).sum().item()
            )

def update_lr_domain_metrics(state: Dict[str, Any], lr_domain_preds: torch.Tensor, regions: torch.Tensor) -> None:
    """Buffer this batch's LR-branch domain predictions/targets for later reduction.

    Appends CPU copies of the predictions and (already-remapped) targets to
    `state`, so a single accuracy computation can run over the whole split at
    epoch end (see `compute_final_lr_domain_metrics`).

    Args:
        state (Dict[str, Any]): Evaluation state dict, as created by
            `make_eval_state`.
        lr_domain_preds (torch.Tensor): Shape `(batch_size,)`, long, argmax
            LR-branch domain-class predictions.
        regions (torch.Tensor): Shape `(batch_size,)`, long. Despite the name,
            this is the already-remapped domain-class target (see
            `build_domain_remap`), not a raw region code -- matches the call
            sites in `multi_scale_classification.py`, which pass
            `domain_targets`.
    """
    state["lr_domain_preds"].append(lr_domain_preds.cpu())
    state["lr_domain_targets"].append(regions.cpu())


def update_hr_domain_metrics(state: Dict[str, Any], hr_domain_preds: torch.Tensor, regions: torch.Tensor) -> None:
    """Buffer this batch's HR-branch domain predictions/targets for later reduction.

    Appends CPU copies of the predictions and (already-remapped) targets to
    `state`, so a single accuracy computation can run over the whole split at
    epoch end (see `compute_final_hr_domain_metrics`).

    Args:
        state (Dict[str, Any]): Evaluation state dict, as created by
            `make_eval_state`.
        hr_domain_preds (torch.Tensor): Shape `(batch_size,)`, long, argmax
            HR-branch domain-class predictions.
        regions (torch.Tensor): Shape `(batch_size,)`, long. Despite the name,
            this is the already-remapped domain-class target (see
            `build_domain_remap`), not a raw region code -- matches the call
            sites in `multi_scale_classification.py`, which pass
            `domain_targets`.
    """
    state["hr_domain_preds"].append(hr_domain_preds.cpu())
    state["hr_domain_targets"].append(regions.cpu())


def compute_final_lr_domain_metrics(state: Dict[str, Any], region_names: List[str]) -> Dict[str, float]:
    """Compute overall and per-region LR-branch domain-classification accuracy.

    Reduces the predictions/targets accumulated in `state` by
    `update_lr_domain_metrics`. Only regions whose name is in `DOMAIN_NAMES`
    get an individual metric.

    Args:
        region_names (List[str]): Despite the parameter name, this is actually
            the *domain*-classifier label space (i.e. `domain_names`, as
            returned by `domain_label_names`), indexed by domain-class id --
            not the raw WILDS region-name list indexed by raw region code.
            See the call site in `compute_final_eval_metrics`, which passes
            `domain_names` here. The naming is inherited from the existing
            code and kept as-is rather than silently renamed.

    Returns:
        Dict[str, float]: `lr-domain-acc-<region>` per reported region plus
            overall `lr-domain-acc`.
    """
    preds = torch.cat(state["lr_domain_preds"])
    targets = torch.cat(state["lr_domain_targets"])
    metrics: Dict[str, float] = {}
    for rid, name in enumerate(region_names):
        if name not in DOMAIN_NAMES:
            continue
        mask = targets == rid
        if mask.sum() == 0:
            continue
        metrics[f"lr-domain-acc-{name.lower()}"] = (preds[mask] == targets[mask]).float().mean().item()
    metrics["lr-domain-acc"] = (preds == targets).float().mean().item()
    return metrics


def compute_final_hr_domain_metrics(state: Dict[str, Any], region_names: List[str]) -> Dict[str, float]:
    """Compute overall and per-region HR-branch domain-classification accuracy.

    Reduces the predictions/targets accumulated in `state` by
    `update_hr_domain_metrics`. Only regions whose name is in `DOMAIN_NAMES`
    get an individual metric.

    Args:
        region_names (List[str]): Despite the parameter name, this is actually
            the *domain*-classifier label space (i.e. `domain_names`, as
            returned by `domain_label_names`), indexed by domain-class id --
            not the raw WILDS region-name list indexed by raw region code.
            See the call site in `compute_final_eval_metrics`, which passes
            `domain_names` here. The naming is inherited from the existing
            code and kept as-is rather than silently renamed.

    Returns:
        Dict[str, float]: `hr-domain-acc-<region>` per reported region plus
            overall `hr-domain-acc`.
    """
    preds = torch.cat(state["hr_domain_preds"])
    targets = torch.cat(state["hr_domain_targets"])
    metrics: Dict[str, float] = {}
    for rid, name in enumerate(region_names):
        if name not in DOMAIN_NAMES:
            continue
        mask = targets == rid
        if mask.sum() == 0:
            continue
        metrics[f"hr-domain-acc-{name.lower()}"] = (preds[mask] == targets[mask]).float().mean().item()
    metrics["hr-domain-acc"] = (preds == targets).float().mean().item()
    return metrics


def compute_final_task_region_metrics(state: Dict[str, Any], region_names: List[str]) -> Dict[str, float]:
    """Compute per-region task accuracy/top5-accuracy/loss and worst-group accuracy.

    Only regions in `DOMAIN_NAMES` with a nonzero sample count are reported.
    Worst-group accuracy is the minimum per-region accuracy (resp. top5
    accuracy) over the reported regions.

    Args:
        region_names (List[str]): Raw WILDS region names indexed by raw region
            code (0-5), e.g. as returned by `extract_region_names`. Contrast
            with `compute_final_lr_domain_metrics` / `compute_final_hr_domain_metrics`,
            where the same parameter name actually holds the domain-label list.

    Returns:
        Dict[str, float]: `region-<name>-task-acc`, `region-<name>-top5-task-acc`,
            and `region-<name>-task-loss` per reported region, plus
            `worst-group-task-acc` and `worst-group-top5-task-acc`.
    """
    worst_group_task_acc = 1.0
    worst_group_top5_task_acc = 1.0
    per_region_task_acc: Dict[str, float] = {}
    per_region_top5_task_acc: Dict[str, float] = {}
    per_region_task_loss: Dict[str, float] = {}
    for rid, rid_total in state["region_total"].items():
        region_name = region_names[rid]
        if rid_total == 0 or region_name not in DOMAIN_NAMES:
            continue
        per_region_task_acc[region_name] = state["task_region_correct"][rid] / rid_total
        per_region_top5_task_acc[region_name] = state["task_top5_region_correct"][rid] / rid_total
        per_region_task_loss[region_name] = state["task_region_loss_sum"][rid] / rid_total
        if per_region_task_acc[region_name] < worst_group_task_acc:
            worst_group_task_acc = per_region_task_acc[region_name]
        if per_region_top5_task_acc[region_name] < worst_group_top5_task_acc:
            worst_group_top5_task_acc = per_region_top5_task_acc[region_name]

    return {
        **{f"region-{name.lower()}-task-acc": acc for name, acc in per_region_task_acc.items()},
        **{f"region-{name.lower()}-top5-task-acc": acc for name, acc in per_region_top5_task_acc.items()},
        **{f"region-{name.lower()}-task-loss": loss for name, loss in per_region_task_loss.items()},
        "worst-group-task-acc": worst_group_task_acc,
        "worst-group-top5-task-acc": worst_group_top5_task_acc,
    }

def compute_final_task_class_metrics(state: Dict[str, Any], region_names: List[str]) -> Dict[str, float]:
    """Compute per-class top-1 task accuracy, overall and per region.

    Classes (or region/class combinations) with no samples are skipped. Class
    names come from the FMoW `categories` list (`TASK_CLASSES`), whose index
    equals the integer label `y`.

    Args:
        region_names (List[str]): Raw WILDS region names indexed by raw region
            code (0-5), e.g. as returned by `extract_region_names`.

    Returns:
        Dict[str, float]: `class-<name>-task-acc` per class with samples, plus
            `region-<region>-class-<name>-task-acc` per (region, class)
            combination with samples, for regions in `DOMAIN_NAMES`.
    """
    metrics: Dict[str, float] = {}
    for c_int, total in state["task_class_total"].items():
        if total == 0:
            continue
        class_name = TASK_CLASSES[c_int]
        metrics[f"class-{class_name}-task-acc"] = state["task_class_correct"][c_int] / total
    for (rid, c_int), total in state["task_region_class_total"].items():
        if total == 0:
            continue
        region_name = region_names[rid] if rid < len(region_names) else None
        if region_name not in DOMAIN_NAMES:
            continue
        class_name = TASK_CLASSES[c_int]
        metrics[f"region-{region_name.lower()}-class-{class_name}-task-acc"] = (
            state["task_region_class_correct"][(rid, c_int)] / total
        )
    return metrics


def compute_final_branch_ablation_metrics(state: Dict[str, Any], region_names: List[str]) -> Dict[str, float]:
    """Compute overall and per-region ablated-branch task accuracy.

    Covers both the LR-ablated and HR-ablated forward passes (see
    `MultiScaleClassificationModule._branch_ablation_step` and
    `FeatureFusionModel.forward_branch_ablation`), plus worst-group accuracy
    (minimum per-region accuracy over reported regions) for each ablation.

    Args:
        state (Dict[str, Any]): Evaluation state dict accumulated over an
            epoch by `MultiScaleClassificationModule._branch_ablation_step`.
        region_names (List[str]): Raw WILDS region names indexed by raw region
            code (0-5), e.g. as returned by `extract_region_names`.

    Returns:
        Dict[str, float]: `lr-ablated-task-acc` / `hr-ablated-task-acc`
            overall, `region-<name>-lr-ablated-task-acc` /
            `region-<name>-hr-ablated-task-acc` per reported region, and
            `lr-ablated-worst-group-task-acc` / `hr-ablated-worst-group-task-acc`.
            Returns `{}` if `state["total"] == 0`.
    """
    total = state["total"]
    if total == 0:
        return {}
    metrics: Dict[str, float] = {
        "lr-ablated-task-acc": state["lr_ablated_task_correct"] / total,
        "hr-ablated-task-acc": state["hr_ablated_task_correct"] / total,
    }
    for prefix, region_correct_key in [("lr-ablated", "lr_ablated_task_region_correct"), ("hr-ablated", "hr_ablated_task_region_correct")]:
        worst_group = 1.0
        for rid, rid_total in state["region_total"].items():
            name = region_names[rid] if rid < len(region_names) else None
            if rid_total == 0 or name not in DOMAIN_NAMES:
                continue
            acc = state[region_correct_key][rid] / rid_total
            metrics[f"region-{name.lower()}-{prefix}-task-acc"] = acc
            if acc < worst_group:
                worst_group = acc
        metrics[f"{prefix}-worst-group-task-acc"] = worst_group
    return metrics


def update_task_metrics(
    state: Dict[str, Any],
    task_criterion_per_sample: torch.nn.Module,
    task_ece_metric: MulticlassCalibrationError,
    task_logits: torch.Tensor,
    task_preds: torch.Tensor,
    y: torch.Tensor,
    metadata: torch.Tensor,
    region_index: int,
) -> None:
    """Accumulate task loss/accuracy/entropy/region/class metrics for one batch.

    Computes per-sample loss and top-1/top-5 correctness, then delegates to
    `update_task_entropy_metrics`, `update_task_region_metrics`, and
    `update_task_class_metrics` to update the corresponding counters in
    `state`, and updates `task_ece_metric` in place.

    Args:
        state (Dict[str, Any]): Evaluation state dict, as created by
            `make_eval_state`.
        task_criterion_per_sample (torch.nn.Module): Per-sample (unreduced)
            cross-entropy loss module.
        task_ece_metric (MulticlassCalibrationError): torchmetrics ECE
            accumulator, updated in place via `.update(probs, y)`.
        task_logits (torch.Tensor): Shape `(batch_size, num_task_labels)`, float.
        task_preds (torch.Tensor): Shape `(batch_size,)`, long, argmax of
            `task_logits`.
        y (torch.Tensor): Shape `(batch_size,)`, long, ground-truth class labels.
        metadata (torch.Tensor): Shape `(batch_size, 6)`, float32, columns
            `[region, year, y, lat, lon, img_span_km]`.
        region_index (int): Column index of the region field in `metadata`.
    """
    probs = torch.softmax(task_logits, dim=1)
    per_sample_loss = task_criterion_per_sample(task_logits, y)
    correct_mask = task_preds == y
    top5_correct_mask = (
        task_logits.topk(TOPK, dim=1).indices == y.unsqueeze(1)
    ).any(dim=1)

    # Accumulate the per-sample loss sum (not the batch mean) so the final task
    # loss is sample-weighted and therefore invariant to batch size / batching.
    state["task_loss_sum"] += per_sample_loss.sum().item()
    state["task_correct"] += correct_mask.sum().item()
    state["task_top5_correct"] += int(top5_correct_mask.sum().item())

    update_task_entropy_metrics(state, probs, correct_mask)
    update_task_region_metrics(state, metadata, region_index, correct_mask, top5_correct_mask, per_sample_loss)
    update_task_class_metrics(state, y, metadata, region_index, correct_mask)
    task_ece_metric.update(probs, y)

def compute_final_task_metrics(state: Dict[str, Any], region_names: List[str], ece_metric: MulticlassCalibrationError) -> Dict[str, float]:
    """Merge task loss/accuracy, calibration, entropy, and region metrics into one dict.

    Task loss/accuracy/top5-accuracy are sample-weighted (not a batch-mean of
    means), since `update_task_metrics` accumulates the per-sample loss as a
    sum in `state["task_loss_sum"]` rather than averaging per batch.

    Args:
        state (Dict[str, Any]): Evaluation state dict accumulated over an
            epoch by `update_task_metrics`.
        region_names (List[str]): Raw WILDS region names indexed by raw region
            code (0-5), passed through to `compute_final_task_region_metrics`.
        ece_metric (MulticlassCalibrationError): torchmetrics ECE accumulator
            updated over the epoch; reduced here via `.compute()`.

    Returns:
        Dict[str, float]: `task-loss`, `task-acc`, `top5-task-acc`, `task-ece`,
            plus everything from `compute_final_task_entropy_metrics` and
            `compute_final_task_region_metrics`.
    """
    total = state["total"]
    task_loss = state["task_loss_sum"] / total if total > 0 else torch.nan
    task_acc = state["task_correct"] / total if total > 0 else torch.nan
    task_top5_acc = state["task_top5_correct"] / total if total > 0 else torch.nan
    task_ece = ece_metric.compute().item()

    task_entropy_metrics = compute_final_task_entropy_metrics(state)
    task_region_metrics = compute_final_task_region_metrics(state, region_names)

    return {
        "task-loss": task_loss,
        "task-acc": task_acc,
        "top5-task-acc": task_top5_acc,
        "task-ece": task_ece,
        **task_entropy_metrics,
        **task_region_metrics,
    }

def update_eval_metrics(
    state: Dict[str, Any],
    task_criterion_per_sample: torch.nn.Module,
    task_ece_metric: MulticlassCalibrationError,
    task_logits: torch.Tensor,
    task_preds: torch.Tensor,
    y: torch.Tensor,
    metadata: torch.Tensor,
    region_index: int,
) -> None:
    """Top-level per-batch entry point: update batch/sample counts, then task metrics.

    Increments `state["batch_count"]` and `state["total"]`, then delegates to
    `update_task_metrics` for the rest of the accumulation.

    Args:
        state (Dict[str, Any]): Evaluation state dict, as created by
            `make_eval_state`.
        task_criterion_per_sample (torch.nn.Module): Per-sample (unreduced)
            cross-entropy loss module.
        task_ece_metric (MulticlassCalibrationError): torchmetrics ECE
            accumulator, updated in place.
        task_logits (torch.Tensor): Shape `(batch_size, num_task_labels)`, float.
        task_preds (torch.Tensor): Shape `(batch_size,)`, long, argmax of
            `task_logits`.
        y (torch.Tensor): Shape `(batch_size,)`, long, ground-truth class labels.
        metadata (torch.Tensor): Shape `(batch_size, 6)`, float32, columns
            `[region, year, y, lat, lon, img_span_km]`.
        region_index (int): Column index of the region field in `metadata`.
    """

    state["batch_count"] += 1
    state["total"] += y.size(0)

    update_task_metrics(state, task_criterion_per_sample, task_ece_metric, task_logits, task_preds, y, metadata, region_index)


def compute_final_eval_metrics(
    state: Dict[str, Any],
    loader_name: str,
    region_names: List[str],
    ece_metric: MulticlassCalibrationError,
    domain_names: List[str],
    include_class_breakdown: bool = False,
) -> Dict[str, float]:
    """Compute final metrics for one dataloader and prefix keys with its name.

    Top-level per-dataloader reduction: wraps `compute_final_task_metrics`
    (and, conditionally, `compute_final_task_class_metrics`,
    `compute_final_lr_domain_metrics`, `compute_final_hr_domain_metrics`), and
    prefixes every metric key with `f"{loader_name}-"` for logging.

    Args:
        state (Dict[str, Any]): Evaluation state dict accumulated over an
            epoch by `update_eval_metrics` (and, if applicable,
            `update_lr_domain_metrics` / `update_hr_domain_metrics`).
        loader_name (str): Name of the dataloader this state belongs to; used
            as the metric key prefix.
        region_names (List[str]): Full WILDS region set, used to name the
            task/region metrics. Kept distinct from `domain_names` so e.g.
            Asia is still reported at test time under Leave-Asia-Out.
        ece_metric (MulticlassCalibrationError): torchmetrics ECE accumulator
            for this dataloader, reduced via `.compute()`.
        domain_names (List[str]): Label space of the domain classifier, used
            to name the domain-accuracy metrics. The full region set in the
            default setting; the smaller trained-region space under
            Leave-Asia-Out, matching the remapped domain targets stored in
            `state`.
        include_class_breakdown (bool): If True, also emit the
            high-cardinality per-class (overall and per-region) task
            accuracies. Used at test time only. Defaults to False.

    Returns:
        Dict[str, float]: All task/region/class/domain metrics for this
            dataloader, each key prefixed with `f"{loader_name}-"`.
    """
    metrics = {}
    metrics.update({
        f"{loader_name}-{k}": v for k, v in compute_final_task_metrics(state, region_names, ece_metric).items()
    })
    if include_class_breakdown:
        metrics.update({
            f"{loader_name}-{k}": v for k, v in compute_final_task_class_metrics(state, region_names).items()
        })
    if state["lr_domain_preds"]:  # Empty, if no LR domain classifier
        metrics.update({
            f"{loader_name}-{k}": v for k, v in compute_final_lr_domain_metrics(state, domain_names).items()
        })
    if state["hr_domain_preds"]:
        metrics.update({
            f"{loader_name}-{k}": v for k, v in compute_final_hr_domain_metrics(state, domain_names).items()
        })

    return metrics
