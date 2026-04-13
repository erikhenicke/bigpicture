from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from lightning import LightningModule
from torch import nn
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import MulticlassCalibrationError
from torchmetrics.classification.accuracy import Accuracy

from models.components.fusion import Fusion
from models.components.branches import DualBranch
from models.utils import make_eval_state, extract_region_names, update_eval_metrics, compute_final_eval_metrics

FIVE_REGIONS = {"Europe", "Americas", "Asia", "Africa", "Oceania"}


class LateFusionModule(LightningModule):
    """Class for late fusion of satellite image features for FMoW classification."""

    def __init__(
        self,
        branches: DualBranch,
        fusion: Fusion,
        optimizer: Callable[..., torch.optim.Optimizer],
        scheduler: Optional[Callable[..., torch.optim.lr_scheduler.LRScheduler]] = None,
        num_labels: int = 62,
        region_index: int = 0,
        ece_n_bins: int = 10,
        val_loader_names: List[str] = ["val"],
        test_loader_names: List[str] = ["test"], 
        key_metric: str = "val/val-od-worst-group-task-acc",
        compile: bool = False,
    ) -> None:
        """Initialize a `LateFusionModule`.

        :param branches: The branches for high and low resolution feature extraction.
        :param fusion: The fusion module.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False,
            ignore=["branches", "fusion", "optimizer", "scheduler"],
        )

        # Keep callables out of checkpoint hyperparameters (PyTorch 2.6+ weights_only default).
        self.optimizer_factory = optimizer
        self.scheduler_factory = scheduler

        self.branches = branches
        self.fusion = fusion
        self.classifier = nn.Linear(self.fusion.out_dim, self.hparams.num_labels)

        self.criterion = nn.CrossEntropyLoss()
        self.criterion_per_sample = nn.CrossEntropyLoss(reduction="none")

        self.train_task_acc = Accuracy(task="multiclass", num_classes=self.hparams.num_labels)
        self.train_task_loss = MeanMetric()
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


    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a forward pass through the model `self.branches`, `self.fusion`, and `self.classifier`.

        :param x: A dict of images.
        :return: A tensor of logits.
        """
        hr_features, lr_features = self.branches(x)
        fused_features = self.fusion(hr_features, lr_features)
        return self.classifier(fused_features)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        self.train()
        self.val_acc_best.reset()

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
        loss = self.criterion(logits, y)
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
        y = batch[1]
        task_loss, task_preds, _ = self.model_step(batch)

        # update and log metrics
        self.train_task_loss(task_loss)
        self.train_task_acc(task_preds, y)
        self.log("train/train-task-loss", self.train_task_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/train-task-acc", self.train_task_acc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return task_loss

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        pass

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
            self.criterion_per_sample,
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
            self.criterion_per_sample,
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
            self.branches = torch.compile(self.branches)
            self.fusion = torch.compile(self.fusion)
            self.classifier = torch.compile(self.classifier)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.optimizer_factory(params=self.parameters())
        if self.scheduler_factory is not None:
            scheduler = self.scheduler_factory(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.hparams.key_metric,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = LateFusionModule(None, None, optimizer=torch.optim.AdamW)
