"""
multi_scale_classification.py

Defines `MultiScaleClassificationModule`, a `lightning.LightningModule`
subclass that is the single fusion-agnostic training/eval harness for every
`MultiScaleModel` architecture in `models.components.fusion_models`
(single-branch, early/feature/decision fusion, D3G). It owns manual
optimization (`automatic_optimization = False`) to support separate
task/LR-domain/HR-domain optimizers, the multi-objective loss (task + LR/HR
domain + D3G consistency), and the shared evaluation suite (ID/OOD splits,
ECE, worst-group, per-class/region, branch ablation, domain confusion
matrices logged to W&B).
"""
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import MulticlassCalibrationError
from torchmetrics.classification.accuracy import Accuracy
import wandb

from models.components.fusion_models import (
    FeatureFusionModel,
    MultiScaleModel,
)
from models.utils import (
    make_eval_state,
    extract_region_names,
    update_eval_metrics,
    update_lr_domain_metrics,
    update_hr_domain_metrics,
    compute_final_eval_metrics,
    compute_final_branch_ablation_metrics,
    build_domain_remap,
    domain_label_names,
)


class MultiScaleClassificationModule(LightningModule):
    """Fusion-agnostic training/eval harness for multi-scale FMoW classification.

    Wraps any model that fulfills the `MultiScaleModel` contract -- single-scale
    (`SingleBranchModel` / `SingleBranchLRModel` / `SingleBranchLocationModel`),
    early fusion (`EarlyFusionModel`), feature fusion (`FeatureFusionModel`),
    domain-gated feature fusion (`D3GModel`), or decision fusion
    (`DecisionFusionModel`) -- and provides the multi-objective training (task +
    LR/HR domain + D3G consistency) and the shared evaluation suite (ID/OOD, ECE,
    worst-group, per-class/region, branch ablation). The wrapper depends only on the
    model's `task_logits` output and `supports_*` capability flags, never on its
    concrete architecture.
    """

    def __init__(
        self,
        model: MultiScaleModel,
        optimizer: Callable[..., torch.optim.Optimizer],
        scheduler: Optional[Callable[..., torch.optim.lr_scheduler.LRScheduler]] = None,
        domain_optimizer: Optional[Callable[..., torch.optim.Optimizer]] = None,
        domain_scheduler: Optional[
            Callable[..., torch.optim.lr_scheduler.LRScheduler]
        ] = None,
        num_task_labels: int = 62,
        num_domain_labels: int = 6,
        domain_index: int = 0,
        ece_n_bins: int = 10,
        val_loader_names: List[str] = ["val"],
        test_loader_names: List[str] = ["test"],
        key_metric: str = "val/val-od-worst-group-task-acc",
        label_smoothing: float = 0.0,
        branch_ablation: bool = False,
        alternating_freeze: bool = False,
        alternating_freeze_period: int = 1,
        leave_asia_out: bool = False,
    ) -> None:
        """Initialize a `MultiScaleClassificationModule`.

        Args:
            model (MultiScaleModel): The wrapped model (any fusion strategy or
                single-branch model); its `supports_*` capability flags
                determine which optimizers/losses/metrics are set up.
            optimizer (Callable[..., torch.optim.Optimizer]): Hydra partial
                factory for the task optimizer; called with
                `params=self.model.task_parameters()`.
            scheduler (Optional[Callable[..., torch.optim.lr_scheduler.LRScheduler]]):
                Hydra partial factory for the task LR scheduler; called with
                `optimizer=<task optimizer>`. If None, no task scheduler is used.
            domain_optimizer (Optional[Callable[..., torch.optim.Optimizer]]):
                Hydra partial factory for the LR-/HR-domain optimizer(s).
                Required (non-None) if the model supports an LR or HR domain
                classifier.
            domain_scheduler (Optional[Callable[..., torch.optim.lr_scheduler.LRScheduler]]):
                Hydra partial factory for the LR-/HR-domain LR scheduler(s).
                If None, no domain scheduler is used.
            num_task_labels (int): Number of task (FMoW category) classes.
                Defaults to 62 (`len(categories)` in WILDS `fmow_dataset.py`).
            num_domain_labels (int): Number of domain-classifier classes. Must
                match `len(domain_label_names(leave_asia_out))`. Defaults to 6.
            domain_index (int): Column index into the `(batch_size, 6)`
                metadata tensor holding the raw WILDS region code. Defaults to 0.
            ece_n_bins (int): Number of bins for the `MulticlassCalibrationError`
                (ECE) metrics. Defaults to 10.
            val_loader_names (List[str]): Names of the validation dataloaders,
                in the order `Trainer.val_dataloaders` provides them; used as
                metric key prefixes and to size `val_ece_metrics`. Defaults to
                `["val"]`.
            test_loader_names (List[str]): Names of the test dataloaders,
                analogous to `val_loader_names`. Defaults to `["test"]`.
            key_metric (str): Metric name monitored by `ReduceLROnPlateau`
                schedulers (task and/or domain). Defaults to
                `"val/val-od-worst-group-task-acc"`.
            label_smoothing (float): Label smoothing passed to the task
                cross-entropy criteria. Defaults to 0.0.
            branch_ablation (bool): If True and the model supports it (see
                `MultiScaleModel.supports_branch_ablation`), additionally run
                the LR-ablated/HR-ablated forward pass during eval and log
                branch-ablation accuracy metrics. Defaults to False.
            alternating_freeze (bool): If True and the model is a
                `FeatureFusionModel`, alternately freeze the LR or HR encoder
                branch across training epochs (see `on_train_epoch_start`).
                Defaults to False.
            alternating_freeze_period (int): Number of consecutive epochs each
                branch stays frozen before alternating, when
                `alternating_freeze` is enabled. Defaults to 1.
            leave_asia_out (bool): If True, use the Leave-Asia-Out domain label
                space (Asia dropped, see `domain_label_names`) and mask Asia
                out of domain metrics/losses. Defaults to False.

        Raises:
            ValueError: If the non-task loss coefficients (LR-domain loss
                coefficient plus D3G consistency loss coefficient, as exposed
                by the model) sum to 1.0 or more, leaving no weight for the
                task loss.
            ValueError: If `num_domain_labels` does not match the size of the
                domain label space implied by `leave_asia_out`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False,
            ignore=[
                "model",
                "optimizer",
                "scheduler",
                "domain_optimizer",
                "domain_scheduler",
            ],
        )

        # Keep callables out of checkpoint hyperparameters (PyTorch 2.6+ weights_only default).
        self.optimizer_factory = optimizer
        self.scheduler_factory = scheduler
        self.domain_optimizer_factory = domain_optimizer
        self.domain_scheduler_factory = domain_scheduler

        self.model = model

        self.use_d3g_objective = self.model.supports_d3g_objective()
        self.has_lr_domain_classifier = self.model.supports_lr_domain_classification()
        self.has_hr_domain_classifier = self.model.supports_hr_domain_classification()

        self.do_branch_ablation = (
            branch_ablation and self.model.supports_branch_ablation()
        )

        non_task_coeff = 0.0
        if self.has_lr_domain_classifier:
            self.lr_domain_loss_coeff = self.model.lr_domain_loss_coeff
            non_task_coeff += self.lr_domain_loss_coeff
        if self.use_d3g_objective:
            self.consistency_loss_coeff = self.model.consistency_loss_coeff
            non_task_coeff += self.consistency_loss_coeff
        if non_task_coeff >= 1.0:
            raise ValueError(
                f"Non-task loss coefficients must sum to less than 1.0, got {non_task_coeff}"
            )
        self.task_loss_coeff = 1.0 - non_task_coeff

        self.task_criterion = nn.CrossEntropyLoss(
            label_smoothing=self.hparams.label_smoothing
        )
        self.task_criterion_per_sample = nn.CrossEntropyLoss(
            reduction="none", label_smoothing=self.hparams.label_smoothing
        )
        # Domain targets are raw WILDS region codes, remapped to the (possibly
        # reduced) domain label space. Under Leave-Asia-Out the only dropped region
        # is Asia, which is absent from training, so the loss never sees the -1
        # sentinel and needs no ignore_index; "Other" remains a normal class.
        self.domain_criterion = nn.CrossEntropyLoss()
        self.domain_names = domain_label_names(self.hparams.leave_asia_out)
        if self.hparams.num_domain_labels != len(self.domain_names):
            raise ValueError(
                f"num_domain_labels ({self.hparams.num_domain_labels}) must match the "
                f"domain label space size ({len(self.domain_names)}) implied by "
                f"leave_asia_out={self.hparams.leave_asia_out}."
            )
        self.register_buffer(
            "domain_remap", build_domain_remap(self.domain_names), persistent=False
        )

        self.train_task_acc = Accuracy(
            task="multiclass", num_classes=self.hparams.num_task_labels
        )
        self.train_task_loss = MeanMetric()

        if self.has_hr_domain_classifier:
            self.train_hr_domain_acc = Accuracy(
                task="multiclass",
                num_classes=self.hparams.num_domain_labels,
                average="none",
            )
            self.train_hr_domain_loss = MeanMetric()

        if self.has_lr_domain_classifier:
            # LR domain metrics
            self.train_lr_domain_acc = Accuracy(
                task="multiclass",
                num_classes=self.hparams.num_domain_labels,
                average="none",
            )
            self.train_lr_domain_loss = MeanMetric()

        self.train_consistency_loss = MeanMetric() if self.use_d3g_objective else None
        self.val_acc_best = MaxMetric()

        self.val_loader_names = val_loader_names
        self.test_loader_names = test_loader_names

        self.val_ece_metrics = nn.ModuleList(
            [
                MulticlassCalibrationError(
                    num_classes=self.hparams.num_task_labels,
                    n_bins=self.hparams.ece_n_bins,
                    norm="l1",
                )
                for _ in range(len(self.val_loader_names))
            ]
        )
        self.test_ece_metrics = nn.ModuleList(
            [
                MulticlassCalibrationError(
                    num_classes=self.hparams.num_task_labels,
                    n_bins=self.hparams.ece_n_bins,
                    norm="l1",
                )
                for _ in range(len(self.test_loader_names))
            ]
        )

        self._val_state: Dict[int, Dict[str, Any]] = {}
        self._test_state: Dict[int, Dict[str, Any]] = {}
        self._val_region_names: Dict[int, List[str]] = {}
        self._test_region_names: Dict[int, List[str]] = {}
        self._task_scheduler = None
        self._lr_domain_scheduler = None
        self._hr_domain_scheduler = None
        self._train_lr_domain_preds: List[torch.Tensor] = []
        self._train_lr_domain_targets: List[torch.Tensor] = []
        if self.has_hr_domain_classifier:
            self._train_hr_domain_preds: List[torch.Tensor] = []
            self._train_hr_domain_targets: List[torch.Tensor] = []

        self.automatic_optimization = False
        self._force_train_mode()

    def _force_train_mode(self) -> None:
        """Force the full module tree into train mode."""
        self.train()
        # Undo the deterministic cuDNN settings applied for eval so training keeps
        # its faster, default (non-deterministic) convolution algorithms.
        torch.backends.cudnn.deterministic = False

    def _force_eval_mode(self) -> None:
        """Force the full module tree into eval mode for deterministic metrics.

        Disables Dropout and switches BatchNorm to its stored running statistics,
        so validation/test forward passes no longer depend on the global RNG state
        or on how samples happen to be batched. The cuDNN flags pin convolution
        algorithms so the logged metrics are reproducible across runs and machines.
        """
        self.eval()
        torch.backends.cudnn.deterministic = True

    def _branch_ablation_step(
        self,
        x: Dict[str, torch.Tensor],
        y: torch.Tensor,
        regions: torch.Tensor,
        state: Dict[str, Any],
    ) -> None:
        """Run the branch-ablation forward pass and accumulate per-branch accuracy.

        Calls `self.model.forward_branch_ablation(x)` (dropping LR or HR
        features via the fusion module) and accumulates overall and
        per-region ablated-branch task accuracy into `state`, in place.

        Args:
            x (Dict[str, torch.Tensor]): Model input dict (see `forward`).
            y (torch.Tensor): Shape `(batch_size,)`, long, ground-truth task
                class labels.
            regions (torch.Tensor): Shape `(batch_size,)`, long, raw WILDS
                region codes.
            state (Dict[str, Any]): Evaluation state dict, as created by
                `models.utils.make_eval_state`.
        """
        result = self.model.forward_branch_ablation(x)
        lr_correct = result["lr_ablated_logits"].argmax(dim=1) == y
        hr_correct = result["hr_ablated_logits"].argmax(dim=1) == y
        state["lr_ablated_task_correct"] += lr_correct.sum().item()
        state["hr_ablated_task_correct"] += hr_correct.sum().item()
        for rid in torch.unique(regions):
            rid_int = int(rid.item())
            mask = regions == rid
            state["lr_ablated_task_region_correct"][rid_int] += int(
                (lr_correct & mask).sum().item()
            )
            state["hr_ablated_task_region_correct"][rid_int] += int(
                (hr_correct & mask).sum().item()
            )

    def forward(
        self,
        x: Dict[str, torch.Tensor],
        region_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Perform a forward pass through the wrapped model, returning task logits only.

        Args:
            x (Dict[str, torch.Tensor]): Model input dict; keys depend on the
                wrapped model, e.g. `"rgb"` shape `(batch_size, 3, H, W)`
                float, `"landsat"` shape `(batch_size, C>=6, H, W)` float,
                optionally `"coord_grid_hr"`/`"coord_grid_lr"` shape
                `(batch_size, 2, H, W)`, `"overlap_mask"` shape
                `(batch_size, 1, H, W)`, `"coords"` shape `(batch_size, 2)`.
            region_ids (Optional[torch.Tensor]): Shape `(batch_size,)`, long,
                raw WILDS region codes (0-5). Only used by `D3GModel`, which
                routes by domain; ignored by other models. Defaults to None.

        Returns:
            torch.Tensor: Shape `(batch_size, num_task_labels)`, float,
                `task_logits`.
        """
        return self._shared_forward(x, region_ids=region_ids)["task_logits"]

    def _shared_forward(
        self,
        x: Dict[str, torch.Tensor],
        region_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Remap region ids into domain-label space, then delegate to the wrapped model.

        Args:
            x (Dict[str, torch.Tensor]): Model input dict (see `forward`).
            region_ids (Optional[torch.Tensor]): Shape `(batch_size,)`, long,
                raw WILDS region codes (0-5). If given, remapped via
                `_domain_targets` before being passed to `self.model`.
                Defaults to None.

        Returns:
            Dict[str, torch.Tensor]: The wrapped model's full output dict.
                Keys vary by capability: `task_logits` always; optionally
                `lr_domain_logits`, `hr_domain_logits`, `rel_logits`,
                depending on which `supports_*` flags the model reports.
        """
        if region_ids is not None:
            # Map raw WILDS region codes to the contiguous domain-label space the
            # model's per-domain heads are indexed in. Identity in the full
            # setting; under Leave-Asia-Out the trained regions shift to 0..N-1
            # and Asia maps to -1 (absent from train/val; at test it only reaches
            # D3G's eval branch, which never indexes by region_ids). Only
            # D3GModel routes by region_ids; other models ignore the argument.
            region_ids = self._domain_targets(region_ids)
        return self.model(x, region_ids=region_ids)

    def on_train_start(self) -> None:
        """Lightning hook called when training begins.

        Resets all train-phase metrics and prediction/target accumulators
        (task, LR-domain, HR-domain, D3G consistency, best validation
        accuracy) at the start of `Trainer.fit`.
        """
        self.val_acc_best.reset()
        self.train_task_loss.reset()
        self.train_task_acc.reset()
        if self.has_hr_domain_classifier:
            self.train_hr_domain_loss.reset()
            self.train_hr_domain_acc.reset()
            self._train_hr_domain_preds = []
            self._train_hr_domain_targets = []
        if self.has_lr_domain_classifier:
            self.train_lr_domain_loss.reset()
            self.train_lr_domain_acc.reset()
            self._train_lr_domain_preds = []
            self._train_lr_domain_targets = []
        if self.use_d3g_objective:
            self.train_consistency_loss.reset()

    def _set_branch_freeze(self, branch: nn.Module, frozen: bool) -> None:
        """Set `requires_grad` on every parameter of `branch`.

        Args:
            branch (nn.Module): Sub-module whose parameters' `requires_grad`
                is set, e.g. `self.model.branches.lr_encoder`.
            frozen (bool): If True, disable gradients for `branch`'s
                parameters (`requires_grad = False`); if False, enable them.
        """
        for param in branch.parameters():
            param.requires_grad = not frozen

    def on_train_epoch_start(self) -> None:
        """Lightning hook called at the start of a training epoch.

        Restores train mode (validation/test leave the module in eval mode).
        Then, only for a `FeatureFusionModel` with `alternating_freeze`
        enabled, unfreezes both branches and re-freezes the LR or HR encoder
        branch on alternating epoch periods (`alternating_freeze_period`),
        logging which branch is frozen.
        """
        # Validation/test put the module into eval mode; restore train mode so the
        # next training epoch runs with Dropout and BatchNorm batch statistics again.
        self._force_train_mode()

        if not self.hparams.alternating_freeze or not isinstance(
            self.model, FeatureFusionModel
        ):
            return

        self._unfreeze_all_branches()

        period = self.hparams.alternating_freeze_period
        if (self.current_epoch // period) % 2 == 0:
            frozen_branch = self.model.branches.lr_encoder
            frozen_name = "lr"
        else:
            frozen_branch = self.model.branches.hr_encoder
            frozen_name = "hr"

        self._set_branch_freeze(frozen_branch, frozen=True)
        self.log(
            "train/frozen_branch",
            float(frozen_name == "lr"),
            on_step=False,
            on_epoch=True,
        )
        self.print(
            f"Epoch {self.current_epoch}: freezing {frozen_name} branch (period={period})"
        )

    def _unfreeze_all_branches(self) -> None:
        """Undo the alternating branch freeze, restoring gradients on both branches.

        No-op unless `alternating_freeze` is enabled and `self.model` is a
        `FeatureFusionModel`.
        """
        if not self.hparams.alternating_freeze or not isinstance(
            self.model, FeatureFusionModel
        ):
            return
        self._set_branch_freeze(self.model.branches.hr_encoder, frozen=False)
        self._set_branch_freeze(self.model.branches.lr_encoder, frozen=False)

    def model_step(
        self, batch: Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the task forward pass and loss for one batch.

        Args:
            batch (Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]):
                `(x, y, metadata)` triple. `x` is the model input dict (see
                `forward`); `y` has shape `(batch_size,)`, long, task class
                labels; `metadata` has shape `(batch_size, 6)`, float32,
                columns `[region, year, y, lat, lon, img_span_km]`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: `(loss, preds,
                logits)`, in that order -- a scalar task cross-entropy loss,
                the shape `(batch_size,)` long argmax predictions, and the
                shape `(batch_size, num_task_labels)` float `task_logits`.
                Note: despite the type hint's parameter-name-agnostic tuple
                and a previous docstring describing this as
                "predictions... targets", the third element returned is the
                raw `task_logits`, not the ground-truth `y`.
        """
        x, y, metadata = batch
        regions = metadata[:, self.hparams.domain_index].long()
        logits = self.forward(x, region_ids=regions)
        loss = self.task_criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, logits

    def _domain_targets(self, regions: torch.Tensor) -> torch.Tensor:
        """Map raw WILDS region codes to the domain-classifier label space.

        Under Leave-Asia-Out only Asia falls outside the label space and maps to
        ``-1``; it is absent from training and is masked out of the domain metrics
        at eval. "Other" is a normal class. With the full region set this is the
        identity map.
        """
        return self.domain_remap.to(regions.device)[regions]

    def log_lr_domain_metrics(
        self,
        lr_domain_loss: torch.Tensor,
        lr_domain_preds: torch.Tensor,
        domain_targets: torch.Tensor,
    ) -> None:
        """Update and log the running train LR-domain loss/accuracy for one batch.

        Updates `train_lr_domain_loss` (`MeanMetric`) and `train_lr_domain_acc`
        (per-class `Accuracy`), logs the epoch-level loss, and buffers CPU
        copies of the predictions/targets for the end-of-epoch confusion
        matrix (see `on_train_epoch_end`).

        Args:
            lr_domain_loss (torch.Tensor): Scalar LR-domain cross-entropy loss
                for this batch.
            lr_domain_preds (torch.Tensor): Shape `(batch_size,)`, long, argmax
                LR-branch domain-class predictions.
            domain_targets (torch.Tensor): Shape `(batch_size,)`, long,
                remapped domain-class targets (see `_domain_targets`).
        """
        self.train_lr_domain_loss(lr_domain_loss)
        self.log(
            "train/train-lr-domain-loss",
            self.train_lr_domain_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.train_lr_domain_acc.update(lr_domain_preds, domain_targets)
        self._train_lr_domain_preds.append(lr_domain_preds.cpu())
        self._train_lr_domain_targets.append(domain_targets.cpu())

    def log_hr_domain_metrics(
        self,
        hr_domain_loss: torch.Tensor,
        hr_domain_preds: torch.Tensor,
        domain_targets: torch.Tensor,
    ) -> None:
        """Update and log the running train HR-domain loss/accuracy for one batch.

        Updates `train_hr_domain_loss` (`MeanMetric`) and `train_hr_domain_acc`
        (per-class `Accuracy`), logs the epoch-level loss, and buffers CPU
        copies of the predictions/targets for the end-of-epoch confusion
        matrix (see `on_train_epoch_end`).

        Args:
            hr_domain_loss (torch.Tensor): Scalar HR-domain cross-entropy loss
                for this batch.
            hr_domain_preds (torch.Tensor): Shape `(batch_size,)`, long, argmax
                HR-branch domain-class predictions.
            domain_targets (torch.Tensor): Shape `(batch_size,)`, long,
                remapped domain-class targets (see `_domain_targets`).
        """
        self.train_hr_domain_loss(hr_domain_loss)
        self.log(
            "train/train-hr-domain-loss",
            self.train_hr_domain_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.train_hr_domain_acc.update(hr_domain_preds, domain_targets)
        self._train_hr_domain_preds.append(hr_domain_preds.cpu())
        self._train_hr_domain_targets.append(domain_targets.cpu())

    def log_task_metrics(
        self, task_loss: torch.Tensor, task_preds: torch.Tensor, y: torch.Tensor
    ) -> None:
        """Update and log the running train task loss/accuracy for one batch.

        Args:
            task_loss (torch.Tensor): Scalar task cross-entropy loss for this
                batch.
            task_preds (torch.Tensor): Shape `(batch_size,)`, long, argmax
                task-class predictions.
            y (torch.Tensor): Shape `(batch_size,)`, long, ground-truth task
                class labels.
        """
        self.train_task_loss(task_loss)
        self.train_task_acc(task_preds, y)
        self.log(
            "train/train-task-loss",
            self.train_task_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/train-task-acc",
            self.train_task_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def training_step(
        self,
        batch: Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Perform one manual-optimization, multi-objective training step.

        Always backpropagates the task loss; additionally the LR-domain loss
        (if the model supports LR-domain classification) and the D3G
        consistency loss (if the model supports the D3G objective) are added
        to a shared `backbone_loss` and backpropagated together via a single
        `manual_backward(backbone_loss)` call. The HR-domain loss (if
        supported) is computed and backpropagated separately afterwards via
        its own `manual_backward(hr_domain_loss)` call, so it does not
        contribute a second, redundant gradient pass through the shared HR
        encoder. Each active optimizer (`task_opt`, and optionally
        `lr_domain_opt`/`hr_domain_opt`) is zeroed before, and stepped after,
        its respective backward pass(es).

        Args:
            batch (Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]):
                `(x, y, metadata)` triple (see `model_step`).
            batch_idx (int): Index of the current batch within the epoch.

        Returns:
            torch.Tensor: The scalar task loss (used by Lightning for logging
                purposes only; gradients are already applied via manual
                optimization above).

        Raises:
            ValueError: If any sub-module of `self.model` is not in training
                mode when this step runs.
            ValueError: If a domain target of -1 (an out-of-label-space
                region, e.g. Asia under Leave-Asia-Out) leaks into the train
                split, for models with an LR- or HR-domain classifier.
        """
        if not all([module.training for module in self.model.modules()]):
            raise ValueError(
                "Model is not in training mode during training step!"
            )

        x, y, metadata = batch
        regions = metadata[:, self.hparams.domain_index].long()

        result = self._shared_forward(x, region_ids=regions)
        task_logits = result["task_logits"]
        task_loss = self.task_criterion(task_logits, y)
        task_preds = torch.argmax(task_logits, dim=1)

        optimizers = self.optimizers()
        # Lightning returns a bare optimizer when there is only one (e.g. decision
        # fusion trained from scratch has just the task optimizer); normalize to a list.
        if not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]
        opt_idx = 0
        task_opt = optimizers[opt_idx]
        opt_idx += 1
        lr_domain_opt = None
        if self.has_lr_domain_classifier:
            lr_domain_opt = optimizers[opt_idx]
            opt_idx += 1
        hr_domain_opt = None
        if self.has_hr_domain_classifier:
            hr_domain_opt = optimizers[opt_idx]

        task_opt.zero_grad()
        if lr_domain_opt is not None:
            lr_domain_opt.zero_grad()
        if hr_domain_opt is not None:
            hr_domain_opt.zero_grad()

        backbone_loss = self.task_loss_coeff * task_loss

        if self.has_lr_domain_classifier or self.has_hr_domain_classifier:
            domain_targets = self._domain_targets(regions)
            if not domain_targets.ge(0).all():
                raise ValueError(
                    "Domain target -1 during training: an out-of-label-space region "
                    "(e.g. Asia under Leave-Asia-Out) leaked into the train split."
                )

        if self.has_lr_domain_classifier:
            lr_domain_logits = result["lr_domain_logits"]
            lr_domain_loss = self.domain_criterion(lr_domain_logits, domain_targets)
            lr_domain_preds = lr_domain_logits.argmax(dim=1)
            self.log_lr_domain_metrics(lr_domain_loss, lr_domain_preds, domain_targets)
            backbone_loss = backbone_loss + self.lr_domain_loss_coeff * lr_domain_loss

        if self.use_d3g_objective:
            d3g_consistency_loss = self.task_criterion(result["rel_logits"], y)
            self.train_consistency_loss(d3g_consistency_loss)
            self.log(
                "train/d3g-consistency-loss",
                d3g_consistency_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
            backbone_loss = (
                backbone_loss + self.consistency_loss_coeff * d3g_consistency_loss
            )

        self.manual_backward(backbone_loss)

        if self.has_hr_domain_classifier:
            hr_domain_logits = result["hr_domain_logits"]
            hr_domain_loss = self.domain_criterion(hr_domain_logits, domain_targets)
            hr_domain_preds = hr_domain_logits.argmax(dim=1)
            self.log_hr_domain_metrics(hr_domain_loss, hr_domain_preds, domain_targets)
            self.manual_backward(hr_domain_loss)

        task_opt.step()
        if lr_domain_opt is not None:
            lr_domain_opt.step()
        if hr_domain_opt is not None:
            hr_domain_opt.step()

        self.log_task_metrics(task_loss, task_preds, y)

        return task_loss

    def _log_domain_confusion_matrix(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        label: str,
        feature_label: str,
        region_names: List[str],
        loader_label: str,
    ) -> None:
        """Log a W&B confusion-matrix plot of domain predictions vs. targets.

        Finds the (first) `WandbLogger` among `self.logger` (which may be a
        single logger or a list of loggers) and logs the plot to its
        `experiment`.

        Args:
            preds (torch.Tensor): Shape `(n,)`, long, domain-class predictions
                for the whole split.
            targets (torch.Tensor): Shape `(n,)`, long, domain-class targets
                for the whole split.
            label (str): W&B log key for the plot, e.g.
                `"train/lr-domain-confusion-matrix"`.
            feature_label (str): Branch label used in the plot title (`"lr"`
                or `"hr"`).
            region_names (List[str]): Domain-class names indexed by
                domain-class id, used as the confusion matrix's class labels.
            loader_label (str): Split/loader label used in the plot title
                (e.g. `"Train"`, or a val/test loader name).

        Raises:
            AttributeError: If no `WandbLogger` is found among `self.logger`
                (`logger` resolves to `None`, and `.experiment` is accessed on
                it).
        """
        logger = self.logger
        if isinstance(logger, list):
            logger = next((log for log in logger if isinstance(log, WandbLogger)), None)
        logger.experiment.log(
            {
                label: wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=targets.tolist(),
                    preds=preds.tolist(),
                    class_names=region_names,
                    title=f"{loader_label} {feature_label.upper()} Domain Confusion Matrix",
                )
            },
            commit=False,
        )

    def on_train_epoch_end(self) -> None:
        """Lightning hook called when a training epoch ends.

        Unfreezes any branch frozen by `alternating_freeze`, then computes and
        logs epoch-level LR/HR domain accuracy (overall and per-class) and W&B
        confusion matrices from the predictions/targets buffered during the
        epoch by `log_lr_domain_metrics` / `log_hr_domain_metrics`.
        """
        self._unfreeze_all_branches()

        # LR domain metrics
        if self._train_lr_domain_preds:
            per_class = self.train_lr_domain_acc.compute()
            for rid, acc in enumerate(per_class):
                name = self.domain_names[rid].lower()
                self.log(f"train/train-lr-domain-acc-{name}", acc)
            preds = torch.cat(self._train_lr_domain_preds)
            targets = torch.cat(self._train_lr_domain_targets)
            self.log("train/train-lr-domain-acc", (preds == targets).float().mean())
            self._log_domain_confusion_matrix(
                preds,
                targets,
                "train/lr-domain-confusion-matrix",
                "lr",
                self.domain_names,
                "Train",
            )

        # HR domain metrics
        if self.has_hr_domain_classifier and self._train_hr_domain_preds:
            per_class = self.train_hr_domain_acc.compute()
            for rid, acc in enumerate(per_class):
                name = self.domain_names[rid].lower()
                self.log(f"train/train-hr-domain-acc-{name}", acc)
            preds = torch.cat(self._train_hr_domain_preds)
            targets = torch.cat(self._train_hr_domain_targets)
            self.log("train/train-hr-domain-acc", (preds == targets).float().mean())
            self._log_domain_confusion_matrix(
                preds,
                targets,
                "train/hr-domain-confusion-matrix",
                "hr",
                self.domain_names,
                "Train",
            )

    def on_validation_epoch_start(self) -> None:
        """Lightning hook that is called at the beginning of a validation epoch.

        1. Initializes the state for each validation dataloader to store metrics.
        2. Resets the ECE metrics for each validation dataloader.
        3. Extracts region names from the validation dataloaders for region-based metrics.

        """
        self._force_eval_mode()
        val_dataloaders = self.trainer.val_dataloaders
        if not isinstance(val_dataloaders, list):
            val_dataloaders = [val_dataloaders]
        self._val_state = {
            idx: make_eval_state() for idx in range(len(val_dataloaders))
        }
        for metric in self.val_ece_metrics:
            metric.reset()
        self._val_region_names = extract_region_names(val_dataloaders)

    def validation_step(
        self,
        batch: Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        Runs the shared forward pass, updates the per-dataloader evaluation
        state (task, LR-/HR-domain, and, if enabled, branch-ablation metrics)
        in place; nothing is returned or logged here (see
        `on_validation_epoch_end` for the reduction/logging).

        Args:
            batch (Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]):
                `(x, y, metadata)` triple (see `model_step`).
            batch_idx (int): Index of the current batch within the dataloader.
            dataloader_idx (int): Index of the validation dataloader this
                batch came from, into `self.val_loader_names` /
                `self.val_ece_metrics` / `self._val_state`. Defaults to 0.

        Raises:
            ValueError: If `dataloader_idx` exceeds the number of configured
                ECE metrics (i.e. more validation dataloaders were provided
                than `val_loader_names` entries at construction time).
            ValueError: If a domain target of -1 (an out-of-label-space
                region, e.g. Asia under Leave-Asia-Out) leaks into a
                validation split, for models with an LR- or HR-domain
                classifier.
        """

        if dataloader_idx >= len(self.val_ece_metrics):
            raise ValueError(
                "Received more validation dataloaders than configured ECE metrics. "
                "Provide matching val_loader_names when initializing MultiScaleClassificationModule."
            )

        x, y, metadata = batch
        regions = metadata[:, self.hparams.domain_index].long()

        result = self._shared_forward(x, region_ids=regions)
        task_logits = result["task_logits"]
        task_preds = torch.argmax(task_logits, dim=1)

        update_eval_metrics(
            self._val_state[dataloader_idx],
            self.task_criterion_per_sample,
            self.val_ece_metrics[dataloader_idx],
            task_logits,
            task_preds,
            y,
            metadata,
            self.hparams.domain_index,
        )

        if self.has_lr_domain_classifier or self.has_hr_domain_classifier:
            # The val splits exclude Asia (LAO drops it), so every sample maps into
            # the domain label space; a -1 would mean an out-of-space region (e.g.
            # Asia) leaked into a val split -- fail loudly rather than skip.
            domain_targets = self._domain_targets(regions)
            if not domain_targets.ge(0).all():
                raise ValueError(
                    "Domain target -1 during validation: an out-of-label-space region "
                    "(e.g. Asia under Leave-Asia-Out) leaked into a val split."
                )

        if self.has_lr_domain_classifier:
            lr_domain_preds = result["lr_domain_logits"].argmax(dim=1)
            update_lr_domain_metrics(
                self._val_state[dataloader_idx], lr_domain_preds, domain_targets
            )
        if self.has_hr_domain_classifier:
            hr_domain_preds = result["hr_domain_logits"].argmax(dim=1)
            update_hr_domain_metrics(
                self._val_state[dataloader_idx], hr_domain_preds, domain_targets
            )

        if self.do_branch_ablation:
            self._branch_ablation_step(x, y, regions, self._val_state[dataloader_idx])

    def on_validation_epoch_end(self) -> None:
        """Lightning hook called when a validation epoch ends.

        Reduces every validation dataloader's accumulated state into logged
        `val/*` metrics via `compute_final_eval_metrics`, and (skipped during
        Lightning's sanity-checking pass) logs domain confusion matrices and,
        if `do_branch_ablation`, branch-ablation metrics for each loader.
        """
        all_metrics: Dict[str, float] = {}
        for idx, state in self._val_state.items():
            loader_name = self.val_loader_names[idx]
            region_names = self._val_region_names.get(idx, [])
            metrics = compute_final_eval_metrics(
                state,
                loader_name,
                region_names,
                self.val_ece_metrics[idx],
                domain_names=self.domain_names,
            )
            all_metrics.update({f"val/{k}": v for k, v in metrics.items()})
            if not self.trainer.sanity_checking:
                if state["lr_domain_preds"]:
                    lr_preds = torch.cat(state["lr_domain_preds"])
                    lr_targets = torch.cat(state["lr_domain_targets"])
                    self._log_domain_confusion_matrix(
                        lr_preds,
                        lr_targets,
                        f"val/{loader_name}-lr-domain-confusion-matrix",
                        "lr",
                        self.domain_names,
                        loader_name,
                    )
                if state["hr_domain_preds"]:
                    hr_preds = torch.cat(state["hr_domain_preds"])
                    hr_targets = torch.cat(state["hr_domain_targets"])
                    self._log_domain_confusion_matrix(
                        hr_preds,
                        hr_targets,
                        f"val/{loader_name}-hr-domain-confusion-matrix",
                        "hr",
                        self.domain_names,
                        loader_name,
                    )
                if self.do_branch_ablation:
                    ablation_metrics = compute_final_branch_ablation_metrics(
                        state, region_names
                    )
                    all_metrics.update(
                        {
                            f"val/{loader_name}-{k}": v
                            for k, v in ablation_metrics.items()
                        }
                    )

        for key, value in all_metrics.items():
            self.log(
                key, value, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True
            )

    def on_test_epoch_start(self) -> None:
        """Lightning hook that is called at the beginning of a test epoch.

        1. Initializes the state for each test dataloader to store metrics.
        2. Resets the ECE metrics for each test dataloader.
        3. Extracts region names from the test dataloaders for region-based metrics.

        """
        self._force_eval_mode()
        test_dataloaders = self.trainer.test_dataloaders
        if not isinstance(test_dataloaders, list):
            test_dataloaders = [test_dataloaders]
        self._test_state = {
            idx: make_eval_state() for idx in range(len(test_dataloaders))
        }
        for metric in self.test_ece_metrics:
            metric.reset()
        self._test_region_names = extract_region_names(test_dataloaders)

    def test_step(
        self,
        batch: Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        Mirrors `validation_step`, with one difference: domain metrics are
        masked to `valid = domain_targets >= 0` rather than asserted
        non-negative, since (unlike validation) a test split can be the
        Asia-only eval split under Leave-Asia-Out, where every domain target
        is -1 and the whole batch is skipped for domain metrics.

        Args:
            batch (Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]):
                `(x, y, metadata)` triple (see `model_step`).
            batch_idx (int): Index of the current batch within the dataloader.
            dataloader_idx (int): Index of the test dataloader this batch came
                from, into `self.test_loader_names` / `self.test_ece_metrics`
                / `self._test_state`. Defaults to 0.

        Raises:
            ValueError: If `dataloader_idx` exceeds the number of configured
                ECE metrics (i.e. more test dataloaders were provided than
                `test_loader_names` entries at construction time).
        """

        if dataloader_idx >= len(self.test_ece_metrics):
            raise ValueError(
                "Received more test dataloaders than configured ECE metrics. "
                "Provide matching test_loader_names when initializing MultiScaleClassificationModule."
            )

        x, y, metadata = batch
        regions = metadata[:, self.hparams.domain_index].long()

        result = self._shared_forward(x, region_ids=regions)
        task_logits = result["task_logits"]
        task_preds = torch.argmax(task_logits, dim=1)

        update_eval_metrics(
            self._test_state[dataloader_idx],
            self.task_criterion_per_sample,
            self.test_ece_metrics[dataloader_idx],
            task_logits,
            task_preds,
            y,
            metadata,
            self.hparams.domain_index,
        )

        if self.has_lr_domain_classifier:
            lr_domain_preds = result["lr_domain_logits"].argmax(dim=1)
            domain_targets = self._domain_targets(regions)
            valid = domain_targets >= 0
            if valid.any():  # all-masked on the Asia-only eval split -> skip
                update_lr_domain_metrics(
                    self._test_state[dataloader_idx],
                    lr_domain_preds[valid],
                    domain_targets[valid],
                )
        if self.has_hr_domain_classifier:
            hr_domain_preds = result["hr_domain_logits"].argmax(dim=1)
            domain_targets = self._domain_targets(regions)
            valid = domain_targets >= 0
            if valid.any():  # all-masked on the Asia-only eval split -> skip
                update_hr_domain_metrics(
                    self._test_state[dataloader_idx],
                    hr_domain_preds[valid],
                    domain_targets[valid],
                )

        if self.do_branch_ablation:
            self._branch_ablation_step(x, y, regions, self._test_state[dataloader_idx])

    def on_test_epoch_end(self) -> None:
        """Lightning hook called when a test epoch ends.

        Mirrors `on_validation_epoch_end` (reduces every test dataloader's
        state into logged `test/*` metrics, logs domain confusion matrices
        and, if `do_branch_ablation`, branch-ablation metrics), except it is
        not skipped during sanity-checking and always passes
        `include_class_breakdown=True` to `compute_final_eval_metrics`, so
        per-class task accuracy is always reported at test time.
        """
        all_metrics: Dict[str, float] = {}
        for idx, state in self._test_state.items():
            loader_name = self.test_loader_names[idx]
            region_names = self._test_region_names.get(idx, [])
            metrics = compute_final_eval_metrics(
                state,
                loader_name,
                region_names,
                self.test_ece_metrics[idx],
                include_class_breakdown=True,
                domain_names=self.domain_names,
            )
            all_metrics.update({f"test/{k}": v for k, v in metrics.items()})
            if state["lr_domain_preds"]:
                lr_preds = torch.cat(state["lr_domain_preds"])
                lr_targets = torch.cat(state["lr_domain_targets"])
                self._log_domain_confusion_matrix(
                    lr_preds,
                    lr_targets,
                    f"test/{loader_name}-lr-domain-confusion-matrix",
                    "lr",
                    self.domain_names,
                    loader_name,
                )
            if state["hr_domain_preds"]:
                hr_preds = torch.cat(state["hr_domain_preds"])
                hr_targets = torch.cat(state["hr_domain_targets"])
                self._log_domain_confusion_matrix(
                    hr_preds,
                    hr_targets,
                    f"test/{loader_name}-hr-domain-confusion-matrix",
                    "hr",
                    self.domain_names,
                    loader_name,
                )
            if self.do_branch_ablation:
                ablation_metrics = compute_final_branch_ablation_metrics(
                    state, region_names
                )
                all_metrics.update(
                    {f"test/{loader_name}-{k}": v for k, v in ablation_metrics.items()}
                )

        for key, value in all_metrics.items():
            self.log(
                key, value, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True
            )

    def configure_optimizers(self) -> Any:
        """Build the task optimizer/scheduler, plus domain optimizer(s)/scheduler(s) if needed.

        Always builds a task optimizer (over `self.model.task_parameters()`)
        and, if `scheduler_factory` is set, a task LR scheduler. Additionally
        builds an LR-domain and/or HR-domain optimizer/scheduler pair (over
        `self.model.lr_domain_parameters()` / `hr_domain_parameters()`) if the
        model supports the corresponding objective, in which case
        `domain_optimizer_factory` must have been provided at construction
        time. `ReduceLROnPlateau` schedulers are additionally configured to
        monitor `self.hparams.key_metric`.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        Returns:
            Any: A list of 1-3 Lightning optimizer/scheduler config dicts (one
                each for task, LR-domain, HR-domain, in that order, omitting
                any objective the model doesn't support).

        Raises:
            ValueError: If the model supports an LR- or HR-domain classifier
                but no `domain_optimizer_factory` was provided at construction
                time.
        """
        task_optimizer = self.optimizer_factory(params=self.model.task_parameters())

        self._task_scheduler = (
            self.scheduler_factory(optimizer=task_optimizer)
            if self.scheduler_factory is not None
            else None
        )

        task_config = {"optimizer": task_optimizer}
        if self._task_scheduler is not None:
            scheduler_config = {
                "scheduler": self._task_scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
            if isinstance(self._task_scheduler, ReduceLROnPlateau):
                scheduler_config["monitor"] = self.hparams.key_metric
            task_config["lr_scheduler"] = scheduler_config

        configs = [task_config]
        self._lr_domain_scheduler = None
        self._hr_domain_scheduler = None

        if (
            self.has_lr_domain_classifier or self.has_hr_domain_classifier
        ) and self.domain_optimizer_factory is None:
            raise ValueError("No domain optimizer factory was provided.")

        if self.has_lr_domain_classifier:
            lr_domain_params = self.model.lr_domain_parameters()
            lr_domain_optimizer = self.domain_optimizer_factory(params=lr_domain_params)

            self._lr_domain_scheduler = (
                self.domain_scheduler_factory(optimizer=lr_domain_optimizer)
                if self.domain_scheduler_factory is not None
                else None
            )

            lr_domain_config = {"optimizer": lr_domain_optimizer}
            if self._lr_domain_scheduler is not None:
                sched_config = {
                    "scheduler": self._lr_domain_scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                }
                if isinstance(self._lr_domain_scheduler, ReduceLROnPlateau):
                    sched_config["monitor"] = self.hparams.key_metric
                lr_domain_config["lr_scheduler"] = sched_config
            configs.append(lr_domain_config)

        if self.has_hr_domain_classifier:
            hr_domain_params = self.model.hr_domain_parameters()
            hr_domain_optimizer = self.domain_optimizer_factory(params=hr_domain_params)

            self._hr_domain_scheduler = (
                self.domain_scheduler_factory(optimizer=hr_domain_optimizer)
                if self.domain_scheduler_factory is not None
                else None
            )

            hr_domain_config = {"optimizer": hr_domain_optimizer}
            if self._hr_domain_scheduler is not None:
                sched_config = {
                    "scheduler": self._hr_domain_scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                }
                if isinstance(self._hr_domain_scheduler, ReduceLROnPlateau):
                    sched_config["monitor"] = self.hparams.key_metric
                hr_domain_config["lr_scheduler"] = sched_config
            configs.append(hr_domain_config)

        return configs


if __name__ == "__main__":
    _ = MultiScaleClassificationModule(None, None, optimizer=torch.optim.AdamW)
