"""
schedulers.py

Learning-rate scheduler(s) used by the training config. Defines
`LinearWarmupCosineAnnealingLR`, an `optim.lr_scheduler.SequentialLR` subclass
that chains a linear warmup phase into cosine annealing; instantiated via
Hydra `_partial_` with the optimizer supplied at runtime by
`MultiScaleClassificationModule.configure_optimizers`.
"""
import torch.optim as optim


class LinearWarmupCosineAnnealingLR(optim.lr_scheduler.SequentialLR):
    """Learning-rate schedule that linearly warms up, then anneals via cosine decay.

    A thin composition of `optim.lr_scheduler.LinearLR` (warmup) and
    `optim.lr_scheduler.CosineAnnealingLR` (decay), stitched together with
    `SequentialLR` so the switch from warmup to annealing happens automatically
    at `warmup_epochs`. Configurable via Hydra `_partial_` (only `optimizer` is
    supplied later, at construction time by the training module).
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        eta_min: float = 1e-6,
        last_epoch: int = -1,
    ) -> None:
        """Initialize the warmup + cosine-annealing schedule.

        Args:
            optimizer (optim.Optimizer): Optimizer whose learning rate is scheduled.
            warmup_epochs (int): Number of epochs over which the learning rate is
                linearly ramped up from `1/warmup_epochs` of the optimizer's base
                LR to the full base LR. If 0, the warmup factor denominator is
                clamped to 1 (i.e. no ramp) via `max(warmup_epochs, 1)`.
            max_epochs (int): Total number of epochs the schedule spans; the cosine
                phase runs for `max_epochs - warmup_epochs` epochs (clamped to at
                least 1).
            eta_min (float): Minimum learning rate reached at the end of cosine
                annealing. Defaults to 1e-6.
            last_epoch (int): Index of the last epoch when resuming a schedule.
                Defaults to -1 (start from scratch).
        """
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
