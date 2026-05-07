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

from models.components.fusion_model import SingleBranchModel, LateFusionModel
from models.utils import make_eval_state, extract_region_names, update_eval_metrics, update_lr_domain_metrics, update_hr_domain_metrics, compute_final_eval_metrics, REGIONS


class LateFusionModule(LightningModule):
    """Class for late fusion of satellite image features for FMoW classification."""

    def __init__(
        self,
        model: LateFusionModel | SingleBranchModel,
        optimizer: Callable[..., torch.optim.Optimizer],
        scheduler: Optional[Callable[..., torch.optim.lr_scheduler.LRScheduler]] = None,
        domain_optimizer: Optional[Callable[..., torch.optim.Optimizer]] = None,
        domain_scheduler: Optional[Callable[..., torch.optim.lr_scheduler.LRScheduler]] = None,
        num_task_labels: int = 62,
        num_domain_labels: int = 6,
        domain_index: int = 0,
        ece_n_bins: int = 10,
        val_loader_names: List[str] = ["val"],
        test_loader_names: List[str] = ["test"],
        key_metric: str = "val/val-od-worst-group-task-acc",
        compile: bool = False,
        label_smoothing: float = 0.0,
    ) -> None:
        """Initialize a `LateFusionModule`.

        :param model: The fusion classifier.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False,
            ignore=["model", "optimizer", "scheduler", "domain_optimizer", "domain_scheduler"],
        )

        # Keep callables out of checkpoint hyperparameters (PyTorch 2.6+ weights_only default).
        self.optimizer_factory = optimizer
        self.scheduler_factory = scheduler
        self.domain_optimizer_factory = domain_optimizer
        self.domain_scheduler_factory = domain_scheduler

        self.model = model

        self.domain_loss_coeff = self.model.domain_loss_coeff

        self.use_d3g_objective = self.model.supports_d3g_objective()
        if self.use_d3g_objective:
            self.d3g_loss_coeff = self.model.consistency_loss_coeff

        self.task_criterion = nn.CrossEntropyLoss(label_smoothing=self.hparams.label_smoothing)
        self.task_criterion_per_sample = nn.CrossEntropyLoss(reduction="none", label_smoothing=self.hparams.label_smoothing)
        self.domain_criterion = nn.CrossEntropyLoss()

        self.train_task_acc = Accuracy(task="multiclass", num_classes=self.hparams.num_task_labels)
        self.train_task_loss = MeanMetric()

        # HR domain metrics (always present)
        self.train_hr_domain_acc = Accuracy(task="multiclass", num_classes=self.hparams.num_domain_labels, average="none")
        self.train_hr_domain_loss = MeanMetric()

        # LR domain metrics (always present)
        self.train_lr_domain_acc = Accuracy(task="multiclass", num_classes=self.hparams.num_domain_labels, average="none")
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
        self._train_hr_domain_preds: List[torch.Tensor] = []
        self._train_hr_domain_targets: List[torch.Tensor] = []

        self.automatic_optimization = False
        self._force_train_mode()

    def _force_train_mode(self) -> None:
        """Force the full module tree into train mode."""
        self.train()
        self.model.train()


    def forward(
        self,
        x: Dict[str, torch.Tensor],
        region_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Perform a forward pass through the model.

        :param x: A dict of images.
        :return: A tensor of logits.
        """
        return self._shared_forward(x, region_ids=region_ids)["task_logits"]


    def _shared_forward(
        self,
        x: Dict[str, torch.Tensor],
        region_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.model(x, region_ids=region_ids)


    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        self.val_acc_best.reset()
        self.train_task_loss.reset()
        self.train_task_acc.reset()
        self.train_hr_domain_loss.reset()
        self.train_hr_domain_acc.reset()
        self._train_hr_domain_preds = []
        self._train_hr_domain_targets = []
        self.train_lr_domain_loss.reset()
        self.train_lr_domain_acc.reset()
        self._train_lr_domain_preds = []
        self._train_lr_domain_targets = []
        if self.use_d3g_objective:
            self.train_consistency_loss.reset()


    def model_step(
        self, batch: Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y, metadata = batch
        regions = metadata[:, self.hparams.domain_index].long()
        logits = self.forward(x, region_ids=regions)
        loss = self.task_criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, logits


    def log_lr_domain_metrics(self, lr_domain_loss: torch.Tensor, lr_domain_preds: torch.Tensor, regions: torch.Tensor) -> None:
        self.train_lr_domain_loss(lr_domain_loss)
        self.log("train/train-lr-domain-loss", self.train_lr_domain_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.train_lr_domain_acc.update(lr_domain_preds, regions)
        self._train_lr_domain_preds.append(lr_domain_preds.cpu())
        self._train_lr_domain_targets.append(regions.cpu())


    def log_hr_domain_metrics(self, hr_domain_loss: torch.Tensor, hr_domain_preds: torch.Tensor, regions: torch.Tensor) -> None:
        self.train_hr_domain_loss(hr_domain_loss)
        self.log("train/train-hr-domain-loss", self.train_hr_domain_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.train_hr_domain_acc.update(hr_domain_preds, regions)
        self._train_hr_domain_preds.append(hr_domain_preds.cpu())
        self._train_hr_domain_targets.append(regions.cpu())


    def log_task_metrics(self, task_loss: torch.Tensor, task_preds: torch.Tensor, y: torch.Tensor) -> None:
        self.train_task_loss(task_loss)
        self.train_task_acc(task_preds, y)
        self.log("train/train-task-loss", self.train_task_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/train-task-acc", self.train_task_acc, on_step=False, on_epoch=True, prog_bar=True)


    def training_step(
        self, batch: Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        assert all([module.training for module in self.model.modules()]), "Model is not in training mode during training step!"

        x, y, metadata = batch
        regions = metadata[:, self.hparams.domain_index].long()

        result = self._shared_forward(x, region_ids=regions)
        task_logits = result["task_logits"]
        task_loss = self.task_criterion(task_logits, y)
        task_preds = torch.argmax(task_logits, dim=1)

        # Get optimizers: [task, lr_domain, hr_domain]
        optimizers = self.optimizers()
        task_opt = optimizers[0] if isinstance(optimizers, list) else optimizers
        lr_domain_opt = optimizers[1]
        hr_domain_opt = optimizers[2]

        task_opt.zero_grad()
        lr_domain_opt.zero_grad()
        hr_domain_opt.zero_grad()

        total_loss = task_loss

        # LR domain loss (coeff controls contribution; 0 means no LR domain training)
        lr_domain_logits = result["lr_domain_logits"]
        lr_domain_loss = self.domain_criterion(lr_domain_logits, regions)
        lr_domain_preds = lr_domain_logits.argmax(dim=1)
        self.log_lr_domain_metrics(lr_domain_loss, lr_domain_preds, regions)
        total_loss = total_loss + self.domain_loss_coeff * lr_domain_loss

        # HR domain loss (always)
        hr_domain_logits = result["hr_domain_logits"]
        hr_domain_loss = self.domain_criterion(hr_domain_logits, regions)
        hr_domain_preds = hr_domain_logits.argmax(dim=1)
        self.log_hr_domain_metrics(hr_domain_loss, hr_domain_preds, regions)
        total_loss = total_loss + hr_domain_loss

        # D3G consistency loss
        if self.use_d3g_objective:
            d3g_consistency_loss = self.task_criterion(result["rel_logits"], y)
            self.train_consistency_loss(d3g_consistency_loss)
            self.log("train/d3g-consistency-loss", d3g_consistency_loss, on_step=False, on_epoch=True, prog_bar=False)
            total_loss = total_loss + self.d3g_loss_coeff * d3g_consistency_loss

        self.manual_backward(total_loss)

        task_opt.step()
        lr_domain_opt.step()
        hr_domain_opt.step()

        self.log_task_metrics(task_loss, task_preds, y)

        return task_loss


    def _log_domain_confusion_matrix(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        label: str,
        region_names: List[str],
    ) -> None:
        logger = self.logger
        if isinstance(logger, list):
            logger = next((log for log in logger if isinstance(log, WandbLogger)), None)
        logger.experiment.log(
            {label: wandb.plot.confusion_matrix(
                probs=None,
                y_true=targets.tolist(),
                preds=preds.tolist(),
                class_names=region_names,
            )},
            commit=False,
        )

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        # LR domain metrics
        if self._train_lr_domain_preds:
            per_class = self.train_lr_domain_acc.compute()
            for rid, acc in enumerate(per_class):
                name = REGIONS[rid].lower()
                self.log(f"train/train-lr-domain-acc-{name}", acc)
            preds = torch.cat(self._train_lr_domain_preds)
            targets = torch.cat(self._train_lr_domain_targets)
            self.log("train/train-lr-domain-acc", (preds == targets).float().mean())
            self._log_domain_confusion_matrix(preds, targets, "train/lr-domain-confusion-matrix", list(REGIONS.values()))

        # HR domain metrics (always)
        if self._train_hr_domain_preds:
            per_class = self.train_hr_domain_acc.compute()
            for rid, acc in enumerate(per_class):
                name = REGIONS[rid].lower()
                self.log(f"train/train-hr-domain-acc-{name}", acc)
            preds = torch.cat(self._train_hr_domain_preds)
            targets = torch.cat(self._train_hr_domain_targets)
            self.log("train/train-hr-domain-acc", (preds == targets).float().mean())
            self._log_domain_confusion_matrix(preds, targets, "train/hr-domain-confusion-matrix", list(REGIONS.values()))


    def on_validation_epoch_start(self) -> None:
        """Lightning hook that is called at the beginning of a validation epoch.

        1. Initializes the state for each validation dataloader to store metrics.
        2. Resets the ECE metrics for each validation dataloader.
        3. Extracts region names from the validation dataloaders for region-based metrics.

        """
        val_dataloaders = self.trainer.val_dataloaders
        if not isinstance(val_dataloaders, list):
            val_dataloaders = [val_dataloaders]
        self._val_state = {idx: make_eval_state() for idx in range(len(val_dataloaders))}
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

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """

        if dataloader_idx >= len(self.val_ece_metrics):
            raise ValueError(
                "Received more validation dataloaders than configured ECE metrics. "
                "Provide matching val_loader_names when initializing LateFusionModule."
            )

        x, y, metadata = batch
        regions = metadata[:, self.hparams.domain_index].long()

        result = self._shared_forward(x, region_ids=regions)
        task_logits = result["task_logits"]
        task_loss = self.task_criterion(task_logits, y)
        task_preds = torch.argmax(task_logits, dim=1)

        update_eval_metrics(
            self._val_state[dataloader_idx],
            self.task_criterion_per_sample,
            self.val_ece_metrics[dataloader_idx],
            task_logits,
            task_loss,
            task_preds,
            y,
            metadata,
            self.hparams.domain_index,
        )
        lr_domain_preds = result["lr_domain_logits"].argmax(dim=1)
        update_lr_domain_metrics(self._val_state[dataloader_idx], lr_domain_preds, regions)
        hr_domain_preds = result["hr_domain_logits"].argmax(dim=1)
        update_hr_domain_metrics(self._val_state[dataloader_idx], hr_domain_preds, regions)


    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        all_metrics: Dict[str, float] = {}
        for idx, state in self._val_state.items():
            loader_name = self.val_loader_names[idx]
            region_names = self._val_region_names.get(idx, [])
            metrics = compute_final_eval_metrics(
                state,
                loader_name,
                region_names,
                self.val_ece_metrics[idx],
            )
            all_metrics.update({f"val/{k}": v for k, v in metrics.items()})
            if not self.trainer.sanity_checking:
                if state["lr_domain_preds"]:
                    lr_preds = torch.cat(state["lr_domain_preds"])
                    lr_targets = torch.cat(state["lr_domain_targets"])
                    self._log_domain_confusion_matrix(
                        lr_preds, lr_targets, f"val/{loader_name}-lr-domain-confusion-matrix", list(REGIONS.values())
                    )
                if state["hr_domain_preds"]:
                    hr_preds = torch.cat(state["hr_domain_preds"])
                    hr_targets = torch.cat(state["hr_domain_targets"])
                    self._log_domain_confusion_matrix(
                        hr_preds, hr_targets, f"val/{loader_name}-hr-domain-confusion-matrix", list(REGIONS.values())
                    )

        for key, value in all_metrics.items():
            self.log(key, value, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)


    def on_test_epoch_start(self) -> None:
        """Lightning hook that is called at the beginning of a test epoch.

        1. Initializes the state for each test dataloader to store metrics.
        2. Resets the ECE metrics for each test dataloader.
        3. Extracts region names from the test dataloaders for region-based metrics.

        """
        test_dataloaders = self.trainer.test_dataloaders
        if not isinstance(test_dataloaders, list):
            test_dataloaders = [test_dataloaders]
        self._test_state = {idx: make_eval_state() for idx in range(len(test_dataloaders))}
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

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """

        if dataloader_idx >= len(self.test_ece_metrics):
            raise ValueError(
                "Received more test dataloaders than configured ECE metrics. "
                "Provide matching test_loader_names when initializing LateFusionModule."
            )

        x, y, metadata = batch
        regions = metadata[:, self.hparams.domain_index].long()

        result = self._shared_forward(x, region_ids=regions)
        task_logits = result["task_logits"]
        task_loss = self.task_criterion(task_logits, y)
        task_preds = torch.argmax(task_logits, dim=1)

        update_eval_metrics(
            self._test_state[dataloader_idx],
            self.task_criterion_per_sample,
            self.test_ece_metrics[dataloader_idx],
            task_logits,
            task_loss,
            task_preds,
            y,
            metadata,
            self.hparams.domain_index,
        )
        lr_domain_preds = result["lr_domain_logits"].argmax(dim=1)
        update_lr_domain_metrics(self._test_state[dataloader_idx], lr_domain_preds, regions)
        hr_domain_preds = result["hr_domain_logits"].argmax(dim=1)
        update_hr_domain_metrics(self._test_state[dataloader_idx], hr_domain_preds, regions)


    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        for idx, state in self._test_state.items():
            loader_name = self.test_loader_names[idx]
            region_names = self._test_region_names.get(idx, [])
            metrics = compute_final_eval_metrics(
                state,
                loader_name,
                region_names,
                self.test_ece_metrics[idx],
            )
            for key, value in metrics.items():
                self.log(f"test/{key}", value, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            if state["lr_domain_preds"]:
                lr_preds = torch.cat(state["lr_domain_preds"])
                lr_targets = torch.cat(state["lr_domain_targets"])
                self._log_domain_confusion_matrix(
                    lr_preds, lr_targets, f"test/{loader_name}-lr-domain-confusion-matrix", region_names
                )
            if state["hr_domain_preds"]:
                hr_preds = torch.cat(state["hr_domain_preds"])
                hr_targets = torch.cat(state["hr_domain_targets"])
                self._log_domain_confusion_matrix(
                    hr_preds, hr_targets, f"test/{loader_name}-hr-domain-confusion-matrix", region_names
                )


    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)


    def configure_optimizers(self) -> Any:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A list of dicts containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        task_optimizer = self.optimizer_factory(params=self.model.task_parameters())

        self._task_scheduler = (
            self.scheduler_factory(optimizer=task_optimizer)
            if self.scheduler_factory is not None
            else None
        )

        task_config = {"optimizer": task_optimizer}
        if self._task_scheduler is not None:
            scheduler_config = {"scheduler": self._task_scheduler, "interval": "epoch", "frequency": 1}
            if isinstance(self._task_scheduler, ReduceLROnPlateau):
                scheduler_config["monitor"] = self.hparams.key_metric
            task_config["lr_scheduler"] = scheduler_config

        configs = [task_config]
        self._lr_domain_scheduler = None
        self._hr_domain_scheduler = None

        if self.domain_optimizer_factory is None:
            raise ValueError("No domain optimizer factory was provided.")

        # LR domain optimizer
        lr_domain_params = self.model.lr_domain_parameters()
        lr_domain_optimizer = self.domain_optimizer_factory(params=lr_domain_params)

        self._lr_domain_scheduler = (
            self.domain_scheduler_factory(optimizer=lr_domain_optimizer)
            if self.domain_scheduler_factory is not None
            else None
        )

        lr_domain_config = {"optimizer": lr_domain_optimizer}
        if self._lr_domain_scheduler is not None:
            sched_config = {"scheduler": self._lr_domain_scheduler, "interval": "epoch", "frequency": 1}
            if isinstance(self._lr_domain_scheduler, ReduceLROnPlateau):
                sched_config["monitor"] = self.hparams.key_metric
            lr_domain_config["lr_scheduler"] = sched_config
        configs.append(lr_domain_config)

        # HR domain optimizer
        hr_domain_params = self.model.hr_domain_parameters()
        hr_domain_optimizer = self.domain_optimizer_factory(params=hr_domain_params)

        self._hr_domain_scheduler = (
            self.domain_scheduler_factory(optimizer=hr_domain_optimizer)
            if self.domain_scheduler_factory is not None
            else None
        )

        hr_domain_config = {"optimizer": hr_domain_optimizer}
        if self._hr_domain_scheduler is not None:
            sched_config = {"scheduler": self._hr_domain_scheduler, "interval": "epoch", "frequency": 1}
            if isinstance(self._hr_domain_scheduler, ReduceLROnPlateau):
                sched_config["monitor"] = self.hparams.key_metric
            hr_domain_config["lr_scheduler"] = sched_config
        configs.append(hr_domain_config)

        return configs


if __name__ == "__main__":
    _ = LateFusionModule(None, None, optimizer=torch.optim.AdamW)
