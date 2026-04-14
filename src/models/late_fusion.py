from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from lightning import LightningModule
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import MulticlassCalibrationError
from torchmetrics.classification.accuracy import Accuracy

from models.components.late_fusion_model import LateFusionModel
from models.utils import make_eval_state, extract_region_names, update_eval_metrics, compute_final_eval_metrics

FIVE_REGIONS = {"Europe", "Americas", "Asia", "Africa", "Oceania"}


class LateFusionModule(LightningModule):
    """Class for late fusion of satellite image features for FMoW classification."""

    def __init__(
        self,
        model: LateFusionModel,
        optimizer: Callable[..., torch.optim.Optimizer],
        scheduler: Optional[Callable[..., torch.optim.lr_scheduler.LRScheduler]] = None,
        domain_optimizer: Optional[Callable[..., torch.optim.Optimizer]] = None,
        domain_scheduler: Optional[Callable[..., torch.optim.lr_scheduler.LRScheduler]] = None,
        num_labels: int = 62,
        domain_num_labels: int = 5,
        region_index: int = 0,
        domain_loss_alpha: float = 0.2,
        ece_n_bins: int = 10,
        val_loader_names: List[str] = ["val"],
        test_loader_names: List[str] = ["test"], 
        key_metric: str = "val/val-od-worst-group-task-acc",
        compile: bool = False,
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

        self.task_criterion = nn.CrossEntropyLoss()
        self.task_criterion_per_sample = nn.CrossEntropyLoss(reduction="none")
        self.domain_criterion = nn.CrossEntropyLoss()

        self.train_task_acc = Accuracy(task="multiclass", num_classes=self.hparams.num_labels)
        self.train_task_loss = MeanMetric()
        self.train_domain_acc = Accuracy(task="multiclass", num_classes=self.hparams.domain_num_labels)
        self.train_domain_loss = MeanMetric()
        self.train_total_loss = MeanMetric()
        self.val_acc_best = MaxMetric()

        self.val_loader_names = val_loader_names 
        self.test_loader_names = test_loader_names 

        self.val_ece_metrics = nn.ModuleList(
            [
                MulticlassCalibrationError(
                    num_classes=self.hparams.num_labels,
                    n_bins=self.hparams.ece_n_bins,
                    norm="l1",
                )
                for _ in range(len(self.val_loader_names))
            ]
        )
        self.test_ece_metrics = nn.ModuleList(
            [
                MulticlassCalibrationError(
                    num_classes=self.hparams.num_labels,
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
        self._domain_scheduler = None

        self.automatic_optimization = False


    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a forward pass through the model.

        :param x: A dict of images.
        :return: A tensor of logits.
        """
        return self._shared_forward(x)["task_logits"]

    def _shared_forward(self, x: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
       return self.model(x)
        

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        self.train()
        self.val_acc_best.reset()
        self.train_task_loss.reset()
        self.train_domain_loss.reset()
        self.train_task_acc.reset()
        self.train_domain_acc.reset()

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
        x, y = batch[0], batch[1]
        logits = self.forward(x)
        loss = self.task_criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, logits

    def training_step(
        self, batch: Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        x, y, metadata = batch
        regions = metadata[:, self.hparams.region_index].long()

        result = self._shared_forward(x)
        task_logits = result["task_logits"]
        domain_logits_detached = result["domain_logits_detached"]
        task_loss = self.task_criterion(task_logits, y)
        domain_loss = self.domain_criterion(domain_logits_detached, regions)
        total_loss = task_loss + self.hparams.domain_loss_alpha * domain_loss

        task_preds = torch.argmax(task_logits, dim=1)
        domain_preds = torch.argmax(domain_logits_detached, dim=1)

        task_optimizer, domain_optimizer = self.optimizers()

        task_optimizer.zero_grad()
        self.toggle_optimizer(task_optimizer)
        self.manual_backward(total_loss)
        task_optimizer.step()
        self.untoggle_optimizer(task_optimizer)

        domain_optimizer.zero_grad()
        self.toggle_optimizer(domain_optimizer)
        domain_loss_head = self.domain_criterion(domain_logits_detached, regions)
        self.manual_backward(domain_loss_head)
        domain_optimizer.step()
        self.untoggle_optimizer(domain_optimizer)

        # update and log metrics
        self.train_task_loss(task_loss)
        self.train_task_acc(task_preds, y)
        self.train_domain_loss(domain_loss)
        self.train_domain_acc(domain_preds, regions)
        self.train_total_loss(total_loss)
        self.log("train/train-task-loss", self.train_task_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/train-task-acc", self.train_task_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/train-domain-loss", self.train_domain_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/train-domain-acc", self.train_domain_acc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/train-total-loss", self.train_total_loss, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return total_loss

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        if self._domain_scheduler is not None:
            self._domain_scheduler.step()

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

        y, metadata = batch[1], batch[2]
        task_loss, task_preds, task_logits = self.model_step(batch)

        update_eval_metrics(
            self._val_state[dataloader_idx],
            self.task_criterion_per_sample,
            self.val_ece_metrics[dataloader_idx],
            task_logits,
            task_loss,
            task_preds,
            y,
            metadata,
            self.hparams.region_index
        )

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

        for key, value in all_metrics.items():
            self.log(key, value, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        if self.trainer.sanity_checking:
            return

        if self._task_scheduler is not None:
            monitor_value = self.trainer.callback_metrics.get(self.hparams.key_metric)
            if monitor_value is not None:
                if isinstance(self._task_scheduler, ReduceLROnPlateau):
                    self._task_scheduler.step(monitor_value.item() if hasattr(monitor_value, "item") else monitor_value)
                else:
                    self._task_scheduler.step()

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

        task_loss, task_preds, task_logits = self.model_step(batch)
        y, metadata = batch[1], batch[2]

        update_eval_metrics(
            self._test_state[dataloader_idx],
            self.task_criterion_per_sample,
            self.test_ece_metrics[dataloader_idx],
            task_logits,
            task_loss,
            task_preds,
            y,
            metadata,
            self.hparams.region_index
        )

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

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        task_optimizer = self.optimizer_factory(params=self.model.task_parameters())
        domain_optimizer = self.domain_optimizer_factory(params=self.model.domain_parameters())

        self._task_scheduler = (
            self.scheduler_factory(optimizer=task_optimizer)
            if self.scheduler_factory is not None 
            else None
        )
        self._domain_scheduler = (
            self.domain_scheduler_factory(optimizer=domain_optimizer)
            if self.domain_scheduler_factory is not None
            else None
        )

        return [task_optimizer, domain_optimizer]


if __name__ == "__main__":
    _ = LateFusionModule(None, None, optimizer=torch.optim.AdamW)
