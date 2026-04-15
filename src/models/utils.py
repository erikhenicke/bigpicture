from collections import defaultdict
from typing import Any, Dict, List

import torch
from torchmetrics.classification import MulticlassCalibrationError


FIVE_REGIONS = {"Europe", "Americas", "Asia", "Africa", "Oceania"}

def make_eval_state() -> Dict[str, Any]:
    """
    Create a new evaluation state dictionary for tracking metrics.
    """
    return {
        "total": 0,
        "batch_count": 0,
        "region_total": defaultdict(int),
        "task_loss_sum": 0.0,
        "task_correct": 0,
        "task_entropy_correct": 0.0,
        "task_entropy_incorrect": 0.0,
        "task_region_correct": defaultdict(int),
        "task_region_loss_sum": defaultdict(float),
    }

def extract_region_names(dataloaders: List[torch.utils.data.DataLoader]) -> Dict[int, List[str]]:
    """
    Extracts the region names present in the metadata of each dataloader.   
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
    """
    Update the task-specific entropy metrics in the state dictionary for the current batch.
    """
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
    if correct_mask.any():
        state["task_entropy_correct"] += entropy[correct_mask].sum().item()
    if (~correct_mask).any():
        state["task_entropy_incorrect"] += entropy[~correct_mask].sum().item()

def compute_final_task_entropy_metrics(state: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute the final task-specific entropy metrics from the state dictionary.
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
    per_sample_loss: torch.Tensor,
) -> None:
    """
    Update the task-specific region metrics in the state dictionary for the current batch.
    """
    region_ids = metadata[:, region_index]
    for rid in torch.unique(region_ids):
        rid_int = int(rid.item())
        mask = region_ids == rid
        state["region_total"][rid_int] += int(mask.sum().item())
        state["task_region_correct"][rid_int] += int((correct_mask & mask).sum().item())
        state["task_region_loss_sum"][rid_int] += float(per_sample_loss[mask].sum().item())

def compute_final_task_region_metrics(state: Dict[str, Any], region_names: List[str]) -> Dict[str, float]:
    """
    Compute the final task-specific region metrics from the state dictionary.
    """
    worst_group_acc = 1.0
    per_region_task_acc: Dict[str, float] = {}
    per_region_task_loss: Dict[str, float] = {}
    for rid, rid_total in state["region_total"].items():
        region_name = region_names[rid]
        if rid_total == 0 or region_name not in FIVE_REGIONS:
            continue
        per_region_task_acc[region_name] = state["task_region_correct"][rid] / rid_total
        per_region_task_loss[region_name] = state["task_region_loss_sum"][rid] / rid_total
        if per_region_task_acc[region_name] < worst_group_acc:
            worst_group_task_acc = per_region_task_acc[region_name]
     
    return {
        **{f"region-{name.lower()}-task-acc": acc for name, acc in per_region_task_acc.items()},
        **{f"region-{name.lower()}-task-loss": loss for name, loss in per_region_task_loss.items()},
        "worst-group-task-acc": worst_group_task_acc,
    }

def update_task_metrics(
    state: Dict[str, Any],
    task_criterion_per_sample: torch.nn.Module,
    task_ece_metric: MulticlassCalibrationError,
    task_logits: torch.Tensor,
    task_loss: torch.Tensor,
    task_preds: torch.Tensor,
    y: torch.Tensor,
    metadata: torch.Tensor,
    region_index: int,
) -> None:
    """
    Update the task-specific metrics in the state dictionary.
    """
    probs = torch.softmax(task_logits, dim=1)
    per_sample_loss = task_criterion_per_sample(task_logits, y)
    correct_mask = task_preds == y

    state["task_loss_sum"] += task_loss.item()
    state["task_correct"] += correct_mask.sum().item()

    update_task_entropy_metrics(state, probs, correct_mask)
    update_task_region_metrics(state, metadata, region_index, correct_mask, per_sample_loss)
    task_ece_metric.update(probs, y)

def compute_final_task_metrics(state: Dict[str, Any], region_names: List[str], ece_metric: MulticlassCalibrationError) -> Dict[str, float]:
    """
    Compute the final task-specific metrics from the state dictionary.
    """
    total = state["total"]
    task_loss = state["task_loss_sum"] / state["batch_count"] if state["batch_count"] > 0 else torch.nan
    task_acc = state["task_correct"] / total if total > 0 else torch.nan
    task_ece = ece_metric.compute().item()

    task_entropy_metrics = compute_final_task_entropy_metrics(state)
    task_region_metrics = compute_final_task_region_metrics(state, region_names)

    return {
        "task-loss": task_loss,
        "task-acc": task_acc,
        "task-ece": task_ece,
        **task_entropy_metrics,
        **task_region_metrics,
    }

def update_eval_metrics(
    state: Dict[str, Any],
    task_criterion_per_sample: torch.nn.Module,
    task_ece_metric: MulticlassCalibrationError,
    task_logits: torch.Tensor,
    task_loss: torch.Tensor,
    task_preds: torch.Tensor,
    y: torch.Tensor,
    metadata: torch.Tensor,
    region_index: int,
) -> None:
    """Updates the state dictionary with the metrics values of the current batch.
    """

    state["batch_count"] += 1
    state["total"] += y.size(0)

    update_task_metrics(state, task_criterion_per_sample, task_ece_metric, task_logits, task_loss, task_preds, y, metadata, region_index)


def compute_final_eval_metrics(
    state: Dict[str, Any],
    loader_name: str,
    region_names: List[str],
    ece_metric: MulticlassCalibrationError,
) -> Dict[str, float]:
    """Compute final metrics for the evaluation phase.
    
    Wraps the task-specific final metrics computation and prefixes metric names with the loader name for logging.
    """

    metrics = {
        f"{loader_name}-{k}": v for k, v in compute_final_task_metrics(state, region_names, ece_metric).items()
    }

    return metrics
