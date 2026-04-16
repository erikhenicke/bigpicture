import torch.optim as optim


class LinearWarmupCosineAnnealingLR(optim.lr_scheduler.SequentialLR):
    """Linear warmup followed by cosine annealing, configurable via Hydra _partial_."""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        eta_min: float = 1e-6,
        last_epoch: int = -1,
    ) -> None:
        warmup = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0 / max(warmup_epochs, 1),
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(max_epochs - warmup_epochs, 1),
            eta_min=eta_min,
        )
        super().__init__(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
            last_epoch=last_epoch,
        )
