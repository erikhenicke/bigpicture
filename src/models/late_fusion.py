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

from models.components.late_fusion_model import LateFusionModel 
from models.utils import make_eval_state, extract_region_names, update_eval_metrics, update_domain_metrics, compute_final_eval_metrics, REGIONS


class LateFusionModule(LightningModule):
    """Class for late fusion of satellite image features for FMoW classification."""

    def __init__(
        self,
        model: LateFusionModel,
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

        self.use_domain_objective = self.model.supports_domain_objective() 
        if self.use_domain_objective:
            self.domain_loss_coeff = self.model.domain_loss_coeff

        self.use_d3g_objective = self.model.supports_d3g_objective()
        if self.use_d3g_objective:
            self.d3g_loss_coeff = self.model.consistency_loss_coeff 

        self.task_criterion = nn.CrossEntropyLoss()
        self.task_criterion_per_sample = nn.CrossEntropyLoss(reduction="none")
        self.domain_criterion = nn.CrossEntropyLoss()

        self.train_task_acc = Accuracy(task="multiclass", num_classes=self.hparams.num_task_labels)
        self.train_task_loss = MeanMetric()
        self.train_domain_acc = (
            Accuracy(task="multiclass", num_classes=self.hparams.num_domain_labels, average="none")
            if self.use_domain_objective
            else None
        )
        self.train_domain_loss = MeanMetric() if self.use_domain_objective else None
        self.train_consistency_loss = MeanMetric() if self.use_d3g_objective else None
        self.train_total_loss = MeanMetric()
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
        self._domain_scheduler = None
        self._train_domain_preds: List[torch.Tensor] = []
        self._train_domain_targets: List[torch.Tensor] = []

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
        if self.use_domain_objective:
            self.train_domain_loss.reset()
            self.train_domain_acc.reset()
            self._train_domain_preds = []
            self._train_domain_targets = []
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
    

    def get_domain_loss(self, result: Dict[str, torch.Tensor], regions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        domain_logits = result.get("domain_logits")
        if domain_logits is None:
            raise RuntimeError("Domain objective is enabled but model outputs are missing 'domain_logits'.")
        domain_loss = self.domain_criterion(domain_logits, regions)
        domain_preds = torch.argmax(domain_logits, dim=1)
        return domain_loss, domain_preds
    

    def log_domain_metrics(self, domain_loss: torch.Tensor, domain_preds: torch.Tensor, regions: torch.Tensor) -> None:
        self.train_domain_loss(domain_loss)
        self.log("train/train-domain-loss", self.train_domain_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.train_domain_acc.update(domain_preds, regions)
        self._train_domain_preds.append(domain_preds.cpu())
        self._train_domain_targets.append(regions.cpu())
    
    
    def task_optimizer_step(self, task_optimizer: torch.optim.Optimizer, task_loss: torch.Tensor) -> None:    
        task_optimizer.zero_grad()
        self.toggle_optimizer(task_optimizer)
        self.manual_backward(task_loss)
        task_optimizer.step()
        self.untoggle_optimizer(task_optimizer)


    def domain_backward(
        self,
        domain_optimizer: torch.optim.Optimizer,
        domain_logits_detached: torch.Tensor,
        regions: torch.Tensor,
    ) -> None:
        domain_loss_head = self.domain_criterion(domain_logits_detached, regions)
        self.manual_backward(domain_loss_head)


    def domain_optimizer_step(self, domain_optimizer: torch.optim.Optimizer) -> None:
        # No toggle needed: domain_logits_detached is computed from lr_features.detach(),
        # so its graph is isolated to domain_classifier weights — task params are unreachable.
        domain_optimizer.zero_grad()
        domain_optimizer.step()


    def log_task_metrics(self, task_loss: torch.Tensor, task_preds: torch.Tensor, y: torch.Tensor, total_loss: torch.Tensor) -> None:
        self.train_task_loss(task_loss)
        self.train_task_acc(task_preds, y)
        self.train_total_loss(total_loss)
        self.log("train/train-task-loss", self.train_task_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/train-task-acc", self.train_task_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/train-total-loss", self.train_total_loss, on_step=False, on_epoch=True, prog_bar=True)


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

        optimizers = self.optimizers()
        total_loss = task_loss

        if self.use_domain_objective:
            domain_loss, domain_preds = self.get_domain_loss(result, regions)
            self.log_domain_metrics(domain_loss, domain_preds, regions)
            total_loss = total_loss + self.domain_loss_coeff * domain_loss

        if self.use_d3g_objective:
            d3g_consistency_loss = self.task_criterion(result["rel_logits"], y)
            self.train_consistency_loss(d3g_consistency_loss)
            self.log("train/d3g-consistency-loss", d3g_consistency_loss, on_step=False, on_epoch=True, prog_bar=False)
            total_loss = total_loss + self.d3g_loss_coeff * d3g_consistency_loss

        if self.use_domain_objective:
            self.domain_backward(optimizers[1], result["domain_logits_detached"], regions)
        self.task_optimizer_step(
            optimizers[0] if isinstance(optimizers, list) else optimizers,
            total_loss,
        )
        if self.use_domain_objective:
            self.domain_optimizer_step(optimizers[1])

        self.log_task_metrics(task_loss, task_preds, y, total_loss)

        # return loss or backpropagation will fail
        return total_loss


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
        if self._domain_scheduler is not None:
            self._domain_scheduler.step()
        if self.use_domain_objective:
            per_class = self.train_domain_acc.compute()
            for rid, acc in enumerate(per_class):
                name = REGIONS[rid].lower()
                self.log(f"train/train-domain-acc-{name}", acc)
            self.log("train/train-domain-acc", per_class.mean())
            self.train_domain_acc.reset()
            preds = torch.cat(self._train_domain_preds)
            targets = torch.cat(self._train_domain_targets)
            self._log_domain_confusion_matrix(preds, targets, "train/domain-confusion-matrix", list(REGIONS.values()))
            self._train_domain_preds = []
            self._train_domain_targets = []


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
        if self.use_domain_objective and "domain_logits" in result:
            domain_preds = result["domain_logits"].argmax(dim=1)
            update_domain_metrics(self._val_state[dataloader_idx], domain_preds, regions)


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
            if state["domain_preds"] and not self.trainer.sanity_checking:
                preds = torch.cat(state["domain_preds"])
                targets = torch.cat(state["domain_targets"])
                self._log_domain_confusion_matrix(
                    preds, targets, f"val/{loader_name}-domain-confusion-matrix", list(REGIONS.values())
                )

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
        if self.use_domain_objective:
            domain_preds = result["domain_logits"].argmax(dim=1)
            update_domain_metrics(self._test_state[dataloader_idx], domain_preds, regions)


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
            if state["domain_preds"]:
                preds = torch.cat(state["domain_preds"])
                targets = torch.cat(state["domain_targets"])
                self._log_domain_confusion_matrix(
                    preds, targets, f"test/{loader_name}-domain-confusion-matrix", region_names
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

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        task_optimizer = self.optimizer_factory(params=self.model.task_parameters())

        self._task_scheduler = (
            self.scheduler_factory(optimizer=task_optimizer)
            if self.scheduler_factory is not None 
            else None
        )

        optimizers = [task_optimizer]
        self._domain_scheduler = None
        if self.use_domain_objective:
            if self.domain_optimizer_factory is None:
                raise ValueError("Domain objective is enabled, but no domain optimizer factory was provided.")
            domain_parameters = self.model.domain_parameters()
            if len(domain_parameters) == 0:
                raise ValueError("Domain objective is enabled, but the model does not expose domain parameters.")
            domain_optimizer = self.domain_optimizer_factory(params=domain_parameters)
            optimizers.append(domain_optimizer)
            self._domain_scheduler = (
                self.domain_scheduler_factory(optimizer=domain_optimizer)
                if self.domain_scheduler_factory is not None
                else None
            )

        return optimizers


if __name__ == "__main__":
    _ = LateFusionModule(None, None, optimizer=torch.optim.AdamW)
