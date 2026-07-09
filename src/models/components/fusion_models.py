"""Full multi-scale classification models, each combining branches + a fusion
strategy behind the common :class:`MultiScaleModel` interface.

``models/multi_scale_classification.py``'s ``MultiScaleClassificationModule``
(the Lightning training/eval harness) depends only on this interface -- every
model here returns a dict containing at least ``task_logits`` from ``forward``,
and advertises which auxiliary objectives it supports via the ``supports_*``
methods, so the harness never needs to know which concrete model it is driving.

Single-scale baselines (one encoder, no fusion): :class:`SingleBranchModel`
(HR image), :class:`SingleBranchLRModel` (LR/Landsat image),
:class:`SingleBranchLocationModel` (geographic coordinates).
:class:`EarlyFusionModel` channel-concatenates HR and LR before a single
encoder. :class:`FeatureFusionModel` runs a :class:`~models.components.
branches.DualBranch` (or ``CoordDualBranch`` / ``DomainDualBranch``) and
combines the two branches' features with a :class:`~models.components.
fusion.Fusion` module; :class:`D3GModel` subclasses it to replace the single
fused task classifier with per-domain heads gated by
:class:`~models.components.domain_relations.D3GRelation` ("domain-gated
feature fusion"). :class:`DecisionFusionModel` instead performs late/decision
fusion of two already-trained single-branch classifiers' logits via a
:class:`~models.components.fusion.DecisionRule`, loading their frozen (or
shape-only) weights from prior single-branch training runs.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from pathlib import Path

import torch
import torch.nn as nn

from models.components.branches import Branch, DualBranch, SatCLIPLocationBranch
from models.components.fusion import Fusion, DecisionRule
from models.components.domain_relations import D3GRelation
from results.utils import find_best_checkpoints, find_run_dir


class MultiScaleModel(nn.Module, ABC):
    """Abstract base class every model wrapped by ``MultiScaleClassificationModule``
    derives from. It is itself an ``nn.Module``, so every model is-a ``nn.Module`` and
    is-a ``MultiScaleModel``. The wrapper is fusion-agnostic: it depends only on this
    interface, not on any concrete subclass.

    Subclasses: ``SingleBranchModel`` / ``SingleBranchLRModel`` /
    ``SingleBranchLocationModel`` (single-scale), ``EarlyFusionModel`` (early
    fusion), ``FeatureFusionModel`` (feature fusion), ``D3GModel`` (domain-gated feature
    fusion), and ``DecisionFusionModel`` (decision fusion).

    The abstract methods below must be implemented by every concrete subclass -- an
    incomplete subclass cannot be instantiated (``TypeError`` at construction). Only
    ``train_model`` has a concrete default.

    ``forward`` always returns a dict containing ``task_logits``. The capability-gated
    keys (``lr_domain_logits`` / ``hr_domain_logits`` / ``rel_logits``) and the optional
    members (``forward_branch_ablation``, ``lr_domain_loss_coeff``,
    ``consistency_loss_coeff``) are present only when the matching ``supports_*`` method
    returns True.
    """

    @abstractmethod
    def forward(
        self, x: Dict[str, torch.Tensor], region_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Run the model on a batch and return its output dict.

        Args:
            x (Dict[str, torch.Tensor]): Batch input dict (keys vary by model;
                typically includes ``"rgb"`` and/or ``"landsat"``, and may
                include ``"coords"``, ``"domain"``, coordinate grids, etc. --
                see each concrete model's ``forward``).
            region_ids (Optional[torch.Tensor]): Shape ``(batch_size,)``,
                integer -- ground-truth domain/region ids, already remapped to
                the model's domain-label space by
                ``MultiScaleClassificationModule._shared_forward``. Only
                :class:`D3GModel` uses this; other models accept and ignore it.

        Returns:
            Dict[str, torch.Tensor]: Always contains ``"task_logits"``
            (shape ``(batch_size, num_task_labels)``). May also contain
            ``"lr_domain_logits"`` / ``"hr_domain_logits"`` (each
            ``(batch_size, num_domain_labels)``) and/or ``"rel_logits"``,
            depending on which ``supports_*`` flags are True.
        """
        ...

    def train_model(self) -> bool:
        """Whether this model is learned via ``Trainer.fit``.

        Concrete default: True for every model, inherited by all subclasses. The sole
        exception is ``DecisionFusionModel``, which overrides this to return its own
        ``train_model`` flag -- False means its frozen heads are evaluated test-only,
        True means the decision heads are trained jointly from scratch.

        Returns:
            bool: True if this model should be fit via ``Trainer.fit``.
        """
        return True

    @abstractmethod
    def supports_d3g_objective(self) -> bool:
        """bool: Whether ``forward`` returns ``"rel_logits"`` for the D3G consistency loss."""
        ...

    @abstractmethod
    def supports_lr_domain_classification(self) -> bool:
        """bool: Whether ``forward`` returns ``"lr_domain_logits"``."""
        ...

    @abstractmethod
    def supports_hr_domain_classification(self) -> bool:
        """bool: Whether ``forward`` returns ``"hr_domain_logits"``."""
        ...

    @abstractmethod
    def supports_branch_ablation(self) -> bool:
        """bool: Whether :meth:`forward_branch_ablation` is available on this model."""
        ...

    @abstractmethod
    def task_parameters(self) -> List[torch.nn.Parameter]:
        """List[torch.nn.Parameter]: Parameters optimized by the task-loss optimizer."""
        ...

    @abstractmethod
    def lr_domain_parameters(self) -> List[torch.nn.Parameter]:
        """List[torch.nn.Parameter]: Parameters optimized by the LR-domain-loss optimizer."""
        ...

    @abstractmethod
    def hr_domain_parameters(self) -> List[torch.nn.Parameter]:
        """List[torch.nn.Parameter]: Parameters optimized by the HR-domain-loss optimizer."""
        ...


class SingleBranchModel(MultiScaleModel):
    """HR-only baseline: single encoder + task classifier, no fusion.

    Optionally augments the HR input with spatial encoding (raw coord channels
    and/or Fourier positional encoding), mirroring the HR handling in
    ``DualBranch``. The encoder must already be built with the matching extra
    ``in_channels`` (see ``make_model``).
    """

    def __init__(
        self,
        encoder: Branch,
        num_task_labels: int,
        num_domain_labels: int = 6,
        coord_channels_hr: bool = False,
        hr_spatial_encoding: Optional[nn.Module] = None,
    ):
        """Build the encoder, task classifier, and HR domain classifier.

        Args:
            encoder (Branch): HR image encoder; must already be built with
                ``in_channels`` matching 3 plus any enabled coordinate/PE extras.
            num_task_labels (int): Number of task classes.
            num_domain_labels (int): Number of domain (region) classes for the
                HR domain classifier.
            coord_channels_hr (bool): If True, concatenate ``x["coord_grid_hr"]``
                onto the HR image before encoding.
            hr_spatial_encoding (Optional[nn.Module]): If given (and
                ``x["coord_grid_hr"]`` is present), applied to the HR coordinate
                grid and the result concatenated onto the HR image.
        """
        super().__init__()
        self.encoder = encoder
        self.coord_channels_hr = coord_channels_hr
        self.hr_spatial_encoding = hr_spatial_encoding
        self.task_classifier = nn.Linear(encoder.out_dim, num_task_labels)
        self.hr_domain_classifier = nn.Linear(encoder.out_dim, num_domain_labels)

    def supports_d3g_objective(self) -> bool:
        """bool: False -- no D3G relation heads."""
        return False

    def supports_lr_domain_classification(self) -> bool:
        """bool: False -- no LR branch/domain classifier."""
        return False

    def supports_hr_domain_classification(self) -> bool:
        """bool: True -- ``forward`` returns "hr_domain_logits"."""
        return True

    def supports_branch_ablation(self) -> bool:
        """bool: False -- single-branch models have nothing to ablate."""
        return False

    def forward(self, x: Dict[str, torch.Tensor], region_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Encode the (optionally augmented) HR image and classify it.

        Args:
            x (Dict[str, torch.Tensor]): Must contain ``"rgb"``
                (``(batch_size, 3, H, W)``, float32), and ``"coord_grid_hr"``
                (``(batch_size, 2, H, W)``, float32) if :attr:`coord_channels_hr`
                is True or :attr:`hr_spatial_encoding` is set.
            region_ids (Optional[torch.Tensor]): Unused; present to satisfy the
                :class:`MultiScaleModel` interface.

        Returns:
            Dict[str, torch.Tensor]: ``"task_logits"``
            (``(batch_size, num_task_labels)``) and ``"hr_domain_logits"``
            (``(batch_size, num_domain_labels)``, computed from detached
            features so the domain gradient does not flow into the encoder).
        """
        hr_image = x["rgb"]

        # Independent, config-driven augmentations (mirrors DualBranch HR handling):
        # raw coords and Fourier PE can each be enabled on their own or together.
        extra_hr = []
        if self.coord_channels_hr:
            extra_hr.append(x["coord_grid_hr"])
        if self.hr_spatial_encoding is not None and "coord_grid_hr" in x:
            extra_hr.append(self.hr_spatial_encoding(x["coord_grid_hr"]))
        if extra_hr:
            hr_image = torch.cat([hr_image] + extra_hr, dim=1)

        features = self.encoder(hr_image)
        return {
            "task_logits": self.task_classifier(features),
            "hr_domain_logits": self.hr_domain_classifier(features.detach()),
        }

    def task_parameters(self) -> List[torch.nn.Parameter]:
        """List[torch.nn.Parameter]: Encoder + task classifier (+ spatial encoding, if any)."""
        params = list(self.encoder.parameters()) + list(self.task_classifier.parameters())
        if self.hr_spatial_encoding is not None:
            params += list(self.hr_spatial_encoding.parameters())
        return params

    def lr_domain_parameters(self) -> List[torch.nn.Parameter]:
        """List[torch.nn.Parameter]: Empty -- this model has no LR domain classifier."""
        return []

    def hr_domain_parameters(self) -> List[torch.nn.Parameter]:
        """List[torch.nn.Parameter]: The HR domain classifier's parameters."""
        return list(self.hr_domain_classifier.parameters())


class SingleBranchLRModel(MultiScaleModel):
    """LR-only model: single LR encoder + task classifier + LR domain head."""

    def __init__(
        self,
        encoder: Branch,
        num_task_labels: int,
        num_domain_labels: int = 6,
        lr_domain_loss_coeff: float = 0.1667,
        landsat_channels: int = 6,
    ):
        """Build the encoder, task classifier, and LR domain classifier.

        Args:
            encoder (Branch): LR (Landsat) image encoder; must already be built
                with ``in_channels == landsat_channels``.
            num_task_labels (int): Number of task classes.
            num_domain_labels (int): Number of domain (region) classes for the
                LR domain classifier.
            lr_domain_loss_coeff (float): Weight of the LR domain loss in the
                combined training objective (read by
                ``MultiScaleClassificationModule``, not used directly here).
            landsat_channels (int): Number of leading channels to keep from the
                raw ``x["landsat"]`` tensor before encoding.
        """
        super().__init__()
        self.encoder = encoder
        self.landsat_channels = landsat_channels
        self.lr_domain_loss_coeff = lr_domain_loss_coeff
        self.task_classifier = nn.Linear(encoder.out_dim, num_task_labels)
        self.lr_domain_classifier = nn.Linear(encoder.out_dim, num_domain_labels)

    def supports_d3g_objective(self) -> bool:
        """bool: False -- no D3G relation heads."""
        return False

    def supports_lr_domain_classification(self) -> bool:
        """bool: True -- ``forward`` returns "lr_domain_logits"."""
        return True

    def supports_hr_domain_classification(self) -> bool:
        """bool: False -- no HR branch/domain classifier."""
        return False

    def supports_branch_ablation(self) -> bool:
        """bool: False -- single-branch models have nothing to ablate."""
        return False

    def forward(self, x: Dict[str, torch.Tensor], region_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Encode the (cropped) LR image and classify it.

        Args:
            x (Dict[str, torch.Tensor]): Must contain ``"landsat"``
                (``(batch_size, C, H, W)``, float32); only the first
                :attr:`landsat_channels` channels are used.
            region_ids (Optional[torch.Tensor]): Unused; present to satisfy the
                :class:`MultiScaleModel` interface.

        Returns:
            Dict[str, torch.Tensor]: ``"task_logits"``
            (``(batch_size, num_task_labels)``) and ``"lr_domain_logits"``
            (``(batch_size, num_domain_labels)``).
        """
        features = self.encoder(x["landsat"][:, :self.landsat_channels, :, :])
        return {
            "task_logits": self.task_classifier(features),
            "lr_domain_logits": self.lr_domain_classifier(features),
        }

    def task_parameters(self) -> List[torch.nn.Parameter]:
        """List[torch.nn.Parameter]: Encoder + task classifier."""
        return list(self.encoder.parameters()) + list(self.task_classifier.parameters())

    def lr_domain_parameters(self) -> List[torch.nn.Parameter]:
        """List[torch.nn.Parameter]: The LR domain classifier's parameters."""
        return list(self.lr_domain_classifier.parameters())

    def hr_domain_parameters(self) -> List[torch.nn.Parameter]:
        """List[torch.nn.Parameter]: Empty -- this model has no HR domain classifier."""
        return []


class SingleBranchLocationModel(MultiScaleModel):
    """Location-only model: SatCLIP encoder + task classifier + LR domain head."""

    def __init__(
        self,
        encoder: SatCLIPLocationBranch,
        num_task_labels: int,
        num_domain_labels: int = 6,
        lr_domain_loss_coeff: float = 0.1667,
    ):
        """Build the location encoder, task classifier, and LR domain classifier.

        Args:
            encoder (SatCLIPLocationBranch): Encoder applied to ``x["coords"]``.
            num_task_labels (int): Number of task classes.
            num_domain_labels (int): Number of domain (region) classes for the
                domain classifier.
            lr_domain_loss_coeff (float): Weight of the domain loss in the
                combined training objective (read by
                ``MultiScaleClassificationModule``, not used directly here).
        """
        super().__init__()
        self.encoder = encoder
        self.lr_domain_loss_coeff = lr_domain_loss_coeff
        self.task_classifier = nn.Linear(encoder.out_dim, num_task_labels)
        self.lr_domain_classifier = nn.Linear(encoder.out_dim, num_domain_labels)

    def supports_d3g_objective(self) -> bool:
        """bool: False -- no D3G relation heads."""
        return False

    def supports_lr_domain_classification(self) -> bool:
        """bool: True -- ``forward`` returns "lr_domain_logits"."""
        return True

    def supports_hr_domain_classification(self) -> bool:
        """bool: False -- no HR branch/domain classifier."""
        return False

    def supports_branch_ablation(self) -> bool:
        """bool: False -- single-branch models have nothing to ablate."""
        return False

    def forward(self, x: Dict[str, torch.Tensor], region_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Encode the geographic coordinates and classify them.

        Args:
            x (Dict[str, torch.Tensor]): Must contain ``"coords"``
                (``(batch_size, 2)``; see
                ``models.components.branches.SatCLIPLocationBranch.forward``).
            region_ids (Optional[torch.Tensor]): Unused; present to satisfy the
                :class:`MultiScaleModel` interface.

        Returns:
            Dict[str, torch.Tensor]: ``"task_logits"``
            (``(batch_size, num_task_labels)``) and ``"lr_domain_logits"``
            (``(batch_size, num_domain_labels)``).
        """
        features = self.encoder(x["coords"])
        return {
            "task_logits": self.task_classifier(features),
            "lr_domain_logits": self.lr_domain_classifier(features),
        }

    def task_parameters(self) -> List[torch.nn.Parameter]:
        """List[torch.nn.Parameter]: Encoder + task classifier."""
        return list(self.encoder.parameters()) + list(self.task_classifier.parameters())

    def lr_domain_parameters(self) -> List[torch.nn.Parameter]:
        """List[torch.nn.Parameter]: The domain classifier's parameters."""
        return list(self.lr_domain_classifier.parameters())

    def hr_domain_parameters(self) -> List[torch.nn.Parameter]:
        """List[torch.nn.Parameter]: Empty -- this model has no HR domain classifier."""
        return []

class EarlyFusionModel(MultiScaleModel):
    """Early fusion model: concatenates HR RGB and Landsat along channels, feeds a single encoder."""

    def __init__(
        self,
        encoder: Branch,
        num_task_labels: int,
        num_domain_labels: int = 6,
        lr_domain_loss_coeff: float = 0.1667,
        landsat_channels: int = 6,
    ):
        """Build the shared encoder, task classifier, and LR domain classifier.

        Args:
            encoder (Branch): Encoder applied to the channel-concatenated
                HR + LR image; must already be built with
                ``in_channels == 3 + landsat_channels``.
            num_task_labels (int): Number of task classes.
            num_domain_labels (int): Number of domain (region) classes for the
                domain classifier.
            lr_domain_loss_coeff (float): Weight of the domain loss in the
                combined training objective (read by
                ``MultiScaleClassificationModule``, not used directly here).
            landsat_channels (int): Number of leading channels to keep from the
                raw ``x["landsat"]`` tensor before concatenation.
        """
        super().__init__()
        self.encoder = encoder
        self.landsat_channels = landsat_channels
        self.lr_domain_loss_coeff = lr_domain_loss_coeff
        self.task_classifier = nn.Linear(encoder.out_dim, num_task_labels)
        self.lr_domain_classifier = nn.Linear(encoder.out_dim, num_domain_labels)

    def supports_d3g_objective(self) -> bool:
        """bool: False -- no D3G relation heads."""
        return False

    def supports_lr_domain_classification(self) -> bool:
        """bool: True -- ``forward`` returns "lr_domain_logits"."""
        return True

    def supports_hr_domain_classification(self) -> bool:
        """bool: False -- no separate HR branch/domain classifier (HR and LR share one encoder)."""
        return False

    def supports_branch_ablation(self) -> bool:
        """bool: False -- single-branch/early-fusion models have nothing to ablate."""
        return False

    def forward(self, x: Dict[str, torch.Tensor], region_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Channel-concatenate HR and (cropped) LR images and classify the pair.

        Args:
            x (Dict[str, torch.Tensor]): Must contain ``"rgb"``
                (``(batch_size, 3, H, W)``, float32) and ``"landsat"``
                (``(batch_size, C, H, W)``, float32; only the first
                :attr:`landsat_channels` channels are used). ``H``/``W`` must
                match between the two.
            region_ids (Optional[torch.Tensor]): Unused; present to satisfy the
                :class:`MultiScaleModel` interface.

        Returns:
            Dict[str, torch.Tensor]: ``"task_logits"``
            (``(batch_size, num_task_labels)``) and ``"lr_domain_logits"``
            (``(batch_size, num_domain_labels)``).
        """
        stacked = torch.cat([x["rgb"], x["landsat"][:, :self.landsat_channels, :, :]], dim=1)
        features = self.encoder(stacked)
        return {
            "task_logits": self.task_classifier(features),
            "lr_domain_logits": self.lr_domain_classifier(features),
        }

    def task_parameters(self) -> List[torch.nn.Parameter]:
        """List[torch.nn.Parameter]: Encoder + task classifier."""
        return list(self.encoder.parameters()) + list(self.task_classifier.parameters())

    def lr_domain_parameters(self) -> List[torch.nn.Parameter]:
        """List[torch.nn.Parameter]: The domain classifier's parameters."""
        return list(self.lr_domain_classifier.parameters())

    def hr_domain_parameters(self) -> List[torch.nn.Parameter]:
        """List[torch.nn.Parameter]: Empty -- this model has no HR domain classifier."""
        return []


class DecisionFusionModel(MultiScaleModel):
    """Decision fusion over precomputed single-branch features.

    Holds two single-branch task heads (HR + LR) and a :class:`DecisionRule`. It
    consumes cached encoder features served by ``source="features"`` -- ``x["rgb"]``
    is the HR feature vector and ``x["landsat"]`` the LR feature vector -- applies
    each head to get per-branch logits, and combines them with the decision rule.

    Two modes, selected by ``train_model``:

    - ``train_model=False`` (default): test-only. The heads copy the trained
      single-branch weights via :meth:`load_heads` (mirroring extract_features.py's
      "build once, reload weights per seed" pattern) and are frozen, so the model
      exposes no task parameters and there is nothing to fit.
    - ``train_model=True``: the heads are reinitialized and left trainable, so they
      can be learned jointly from scratch through the decision rule via
      ``Trainer.fit``. Only the input width is taken from the checkpoint; the weights
      are not copied.

    Either way the heads supports neither the domain nor the D3G objectives: the
    domain metrics already live in each single-branch run folder and worst-group
    accuracy uses the ground-truth region, so no domain head is needed here.

    The heads are built in the constructor, which reads each branch's feature width
    straight from the saved classifier weight -- so the feature dimension is never
    configured, it is inferred from the checkpoint. The class prior P(y) is handled
    entirely by the :class:`DecisionRule` (computed from metadata in its constructor).
    """

    def __init__(
        self,
        num_task_labels: int,
        rule: DecisionRule,
        hr_run_name: str,
        lr_run_name: str,
        run_idx: int = 0,
        train_model: bool = False,
    ):
        """Build the decision rule and load/initialize this seed's two heads.

        Args:
            num_task_labels (int): Number of task classes; must match both
                loaded heads' output width.
            rule (DecisionRule): Decision rule combining the two heads' logits.
            hr_run_name (str): Experiment key of the HR single-branch training
                run whose checkpoint supplies the HR head (see
                :meth:`_load_heads`).
            lr_run_name (str): Experiment key of the LR single-branch training
                run whose checkpoint supplies the LR head.
            run_idx (int): Which seed's checkpoint to load from each run
                (indexes ``find_best_checkpoints``'s sorted seed list).
            train_model (bool): If False (default), heads copy the trained
                checkpoint weights and are frozen (test-only). If True, heads
                are reinitialized (only their shape taken from the checkpoint)
                and left trainable. See :meth:`_load_heads`.
        """
        super().__init__()
        self.num_task_labels = num_task_labels
        self.rule = rule
        # Stored under a leading underscore so it never shadows the train_model()
        # method (an instance attribute of the same name would mask the method).
        self._train_model = train_model
        self.hr_head: Optional[nn.Linear] = None
        self.lr_head: Optional[nn.Linear] = None
        self._load_heads(hr_run_name, lr_run_name, run_idx)

    def train_model(self) -> bool:
        """bool: This instance's ``train_model`` flag (see :meth:`__init__`)."""
        return self._train_model

    def supports_d3g_objective(self) -> bool:
        """bool: False -- decision fusion has no D3G relation heads."""
        return False

    def supports_lr_domain_classification(self) -> bool:
        """bool: False -- domain metrics live in the single-branch runs; see class docstring."""
        return False

    def supports_hr_domain_classification(self) -> bool:
        """bool: False -- domain metrics live in the single-branch runs; see class docstring."""
        return False

    def supports_branch_ablation(self) -> bool:
        """bool: False -- nothing to ablate here; each head is already a single branch."""
        return False

    def forward(self, x: Dict[str, torch.Tensor], region_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Apply each frozen/trainable head to its cached features and fuse via the rule.

        Args:
            x (Dict[str, torch.Tensor]): Must contain ``"rgb"``
                (``(batch_size, hr_in_dim)``, float32 -- cached HR encoder
                features, not an image) and ``"landsat"``
                (``(batch_size, lr_in_dim)``, float32 -- cached LR encoder
                features), as served by the dataset in ``source="features"``
                mode. ``hr_in_dim`` / ``lr_in_dim`` are the input widths read
                from the loaded checkpoints (see :meth:`_load_heads`).
            region_ids (Optional[torch.Tensor]): Unused; present to satisfy the
                :class:`MultiScaleModel` interface.

        Returns:
            Dict[str, torch.Tensor]: ``{"task_logits": ...}``, shape
            ``(batch_size, num_task_labels)`` -- the log fused posterior from
            :attr:`rule`.

        Raises:
            RuntimeError: If :attr:`hr_head` or :attr:`lr_head` is ``None``
                (i.e. :meth:`_load_heads` did not run/succeed).
        """
        if self.hr_head is None or self.lr_head is None:
            raise RuntimeError("DecisionFusionModel heads were not built in the constructor.")
        hr_logits = self.hr_head(x["rgb"])
        lr_logits = self.lr_head(x["landsat"])
        return {"task_logits": self.rule(hr_logits, lr_logits)}

    def _load_heads(self, hr_run_name: str, lr_run_name: str, run_idx: int) -> None:
        """Build this seed's two task heads from the single-branch checkpoints.

        The HR/LR heads come from the same single-branch runs that produced the cached
        features, so ``hr_run_name`` / ``lr_run_name`` must be the experiment keys (e.g.
        ``densenet_baseline``, which ``find_run_dir`` resolves to ``train_<key>``). The
        head for seed ``run_idx`` is built and its feature width read straight from the
        saved ``model.task_classifier.*`` weight (the ``model.`` prefix comes from the
        ``MultiScaleClassificationModule`` wrapper), so the feature dimension is never
        configured.

        In test-only mode (``train_model=False``) the heads copy the trained weights and
        are frozen. In training mode (``train_model=True``) only the shape is taken from
        the checkpoint -- the heads are reinitialized and left trainable so they can be
        learned jointly from scratch through the decision rule.

        Args:
            hr_run_name (str): Experiment key resolved via ``find_run_dir`` to
                the HR single-branch run directory.
            lr_run_name (str): Experiment key resolved via ``find_run_dir`` to
                the LR single-branch run directory.
            run_idx (int): Seed index into each run's sorted checkpoint list.

        Returns:
            None: Sets :attr:`hr_head` and :attr:`lr_head` in place.

        Raises:
            FileNotFoundError: If either run name does not resolve to a run
                directory via ``find_run_dir``.
            IndexError: If ``run_idx`` is out of range for either run's
                available checkpoints.
        """
        hr_dir = find_run_dir(hr_run_name)
        lr_dir = find_run_dir(lr_run_name)
        if hr_dir is None or lr_dir is None:
            raise FileNotFoundError(
                f"Could not resolve head runs: hr_run_name={hr_run_name} -> {hr_dir}, "
                f"lr_run_name={lr_run_name} -> {lr_dir}"
            )
        hr_ckpts = find_best_checkpoints(hr_dir)
        lr_ckpts = find_best_checkpoints(lr_dir)
        if run_idx >= len(hr_ckpts) or run_idx >= len(lr_ckpts):
            raise IndexError(
                f"Seed {run_idx} out of range (HR has {len(hr_ckpts)} seeds, LR has {len(lr_ckpts)})."
            )

        def state(path: Path) -> dict:
            """Load a checkpoint file's Lightning ``state_dict``.

            Args:
                path (Path): Checkpoint file path.

            Returns:
                dict: The checkpoint's ``"state_dict"`` entry (parameter name ->
                tensor).
            """
            return torch.load(path, map_location="cpu", weights_only=False)["state_dict"]

        device = self.rule.log_class_prior.device
        copy_weights = not self._train_model
        self.hr_head = self._build_head(state(hr_ckpts[run_idx]), copy_weights=copy_weights).to(device)
        self.lr_head = self._build_head(state(lr_ckpts[run_idx]), copy_weights=copy_weights).to(device)
        if not self._train_model:
            # Frozen: loaded from trained checkpoints, never optimized.
            self.requires_grad_(False)

    def _build_head(self, state_dict: Dict[str, torch.Tensor], copy_weights: bool) -> nn.Linear:
        """Build a single ``Linear`` head sized from a checkpoint's task-classifier weight.

        Args:
            state_dict (Dict[str, torch.Tensor]): A single-branch checkpoint's
                ``state_dict``; must contain ``"model.task_classifier.weight"``
                (shape ``(out_dim, in_dim)``) and ``"...bias"``.
            copy_weights (bool): If True, copy the checkpoint's weight/bias into
                the new head (test-only mode). If False, leave the head at its
                fresh ``nn.Linear`` initialization (training-from-scratch mode).

        Returns:
            nn.Linear: A new ``Linear(in_dim, out_dim)`` head, optionally
            initialized with the checkpoint's weights.

        Raises:
            ValueError: If the checkpoint's output width does not match
                :attr:`num_task_labels`.
        """
        weight = state_dict["model.task_classifier.weight"]
        bias = state_dict["model.task_classifier.bias"]
        out_dim, in_dim = weight.shape
        if out_dim != self.num_task_labels:
            raise ValueError(
                f"Head has {out_dim} outputs but num_task_labels={self.num_task_labels}."
            )
        head = nn.Linear(in_dim, out_dim)
        if copy_weights:
            with torch.no_grad():
                head.weight.copy_(weight)
                head.bias.copy_(bias)
        return head

    def task_parameters(self) -> List[torch.nn.Parameter]:
        """List[torch.nn.Parameter]: Empty if frozen (``train_model=False``); otherwise both heads' parameters."""
        if not self._train_model:
            return []
        params: List[torch.nn.Parameter] = []
        if self.hr_head is not None:
            params += list(self.hr_head.parameters())
        if self.lr_head is not None:
            params += list(self.lr_head.parameters())
        return params

    def lr_domain_parameters(self) -> List[torch.nn.Parameter]:
        """List[torch.nn.Parameter]: Empty -- this model has no domain classifier."""
        return []

    def hr_domain_parameters(self) -> List[torch.nn.Parameter]:
        """List[torch.nn.Parameter]: Empty -- this model has no domain classifier."""
        return []


class FeatureFusionModel(MultiScaleModel):
    """Feature fusion: two branches + a :class:`~models.components.fusion.Fusion`
    module + task/domain classifiers.

    Runs ``branches`` (typically a :class:`~models.components.branches.DualBranch`)
    to get an HR and an LR feature vector, combines them with ``fusion``, and
    feeds the result to a task classifier (or, if the fusion module already
    :attr:`~models.components.fusion.Fusion.produces_logits`, uses its output as
    the logits directly, with no extra classifier). Also carries independent LR
    and HR domain classifiers, each fed from its own branch's *un-fused* features
    (LR through :attr:`detach_lr_for_task` if that would otherwise leak into the
    task gradient; HR always detached before the HR domain head). Subclassed by
    :class:`D3GModel`, which replaces the single shared task classifier with
    per-domain heads (hence :meth:`supports_d3g_objective` checks
    ``isinstance(self, D3GModel)``).
    """

    def __init__(
        self,
        branches: DualBranch,
        fusion: Optional[Fusion],
        num_task_labels: int,
        num_domain_labels: int,
        lr_domain_loss_coeff: float = 0.1667,
        detach_lr_for_task: bool = False,
    ):
        """Build the branches wrapper, task classifier (if needed), and domain classifiers.

        Args:
            branches (DualBranch): Runs the HR and LR encoders; see
                :class:`~models.components.branches.DualBranch`.
            fusion (Optional[Fusion]): Combines the two branches' features. May
                be ``None`` for a subclass (e.g. :class:`D3GModel`) that
                overrides ``forward`` and does not use a shared fusion module;
                the base :meth:`forward` requires it to be set.
            num_task_labels (int): Number of task classes.
            num_domain_labels (int): Number of domain (region) classes for both
                domain classifiers.
            lr_domain_loss_coeff (float): Weight of the LR domain loss in the
                combined training objective (read by
                ``MultiScaleClassificationModule``, not used directly here).
            detach_lr_for_task (bool): If True, detach the LR branch features
                before fusion/task-classification (so no task gradient flows
                into the LR encoder), while still using the un-detached LR
                features for the LR domain classifier.

        Raises:
            AssertionError: If ``fusion`` is given, produces logits directly
                (``fusion.produces_logits``), and its ``out_dim`` does not equal
                ``num_task_labels``.
        """
        super().__init__()

        self.branches: DualBranch = branches
        self.fusion = fusion
        self.lr_domain_loss_coeff = lr_domain_loss_coeff
        self.detach_lr_for_task = detach_lr_for_task
        self.task_classifier: Optional[nn.Linear] = None
        if self.fusion is not None and not self.fusion.produces_logits:
            self.task_classifier = nn.Linear(self.fusion.out_dim, num_task_labels)
        elif self.fusion is not None:
            assert self.fusion.out_dim == num_task_labels, (
                f"Fusion produces logits but out_dim ({self.fusion.out_dim}) != "
                f"num_task_labels ({num_task_labels})"
            )
        self.lr_domain_classifier = nn.Linear(branches.lr_encoder.out_dim, num_domain_labels)
        self.hr_domain_classifier = nn.Linear(branches.hr_encoder.out_dim, num_domain_labels)

    def supports_d3g_objective(self) -> bool:
        """bool: True only for :class:`D3GModel` instances (the D3G consistency loss)."""
        return isinstance(self, D3GModel)

    def supports_lr_domain_classification(self) -> bool:
        """bool: True -- ``forward`` returns "lr_domain_logits"."""
        return True

    def supports_hr_domain_classification(self) -> bool:
        """bool: True -- ``forward`` returns "hr_domain_logits"."""
        return True

    def forward(
        self,
        x: Dict[str, torch.Tensor],
        region_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Encode both branches, fuse them, and classify task + both domains.

        Args:
            x (Dict[str, torch.Tensor]): Batch dict forwarded to
                :attr:`branches` (see
                ``models.components.branches.DualBranch.forward``).
            region_ids (Optional[torch.Tensor]): Unused; present to satisfy the
                :class:`MultiScaleModel` interface.

        Returns:
            Dict[str, torch.Tensor]: ``"task_logits"``
            (``(batch_size, num_task_labels)``), ``"lr_domain_logits"`` and
            ``"hr_domain_logits"`` (each ``(batch_size, num_domain_labels)``).
            The HR domain logits are always computed from detached HR features;
            the LR domain logits use un-detached LR features regardless of
            :attr:`detach_lr_for_task`.

        Raises:
            RuntimeError: If :attr:`fusion` is ``None``.
        """
        if self.fusion is None:
            raise RuntimeError("FeatureFusionModel requires a fusion module for forward().")

        hr_branch_out, lr_branch_out = self.branches(x)
        lr_for_domain = lr_branch_out
        if self.detach_lr_for_task:
            lr_branch_out = lr_branch_out.detach()
        fused = self.fusion(hr_branch_out, lr_branch_out)
        task_logits = self.task_classifier(fused) if self.task_classifier is not None else fused
        outputs = {
            "task_logits": task_logits,
            "lr_domain_logits": self.lr_domain_classifier(lr_for_domain),
            "hr_domain_logits": self.hr_domain_classifier(hr_branch_out.detach()),
        }
        return outputs

    def supports_branch_ablation(self) -> bool:
        """bool: True if a fusion module is set (branch ablation needs ``Fusion.forward_branch_ablation``)."""
        return self.fusion is not None

    def forward_branch_ablation(
        self, x: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Run the fusion module with each branch ablated in turn, to isolate its contribution.

        Args:
            x (Dict[str, torch.Tensor]): Batch dict forwarded to
                :attr:`branches`.

        Returns:
            Dict[str, torch.Tensor]: ``"lr_ablated_logits"`` (LR branch zeroed
            out) and ``"hr_ablated_logits"`` (HR branch zeroed out), each shape
            ``(batch_size, num_task_labels)`` if :attr:`task_classifier` is set,
            else ``(batch_size, fusion.out_dim)``.
        """
        hr_features, lr_features = self.branches(x)
        lr_ablated = self.fusion.forward_branch_ablation(hr_features, lr_features, cutoff="lr")
        hr_ablated = self.fusion.forward_branch_ablation(hr_features, lr_features, cutoff="hr")
        if self.task_classifier is not None:
            lr_ablated = self.task_classifier(lr_ablated)
            hr_ablated = self.task_classifier(hr_ablated)
        return {
            "lr_ablated_logits": lr_ablated,
            "hr_ablated_logits": hr_ablated,
        }

    def task_parameters(self) -> List[torch.nn.Parameter]:
        """List[torch.nn.Parameter]: Branches + fusion (if any) + task classifier (if any)."""
        parameters = list(self.branches.parameters())
        if self.fusion is not None:
            parameters += list(self.fusion.parameters())
        if self.task_classifier is not None:
            parameters += list(self.task_classifier.parameters())
        return parameters

    def lr_domain_parameters(self) -> List[torch.nn.Parameter]:
        """List[torch.nn.Parameter]: The LR domain classifier's parameters."""
        return list(self.lr_domain_classifier.parameters())

    def hr_domain_parameters(self) -> List[torch.nn.Parameter]:
        """List[torch.nn.Parameter]: The HR domain classifier's parameters."""
        return list(self.hr_domain_classifier.parameters())


class D3GModel(FeatureFusionModel):
    """D3G: domain-gated feature fusion with one task-classifier head per domain.

    Instead of a single shared fused-feature classifier, keeps one ``nn.Linear``
    task head per domain (``task_classifiers``, indexed 0..``num_domain_labels``
    - 1), all applied to the (un-fused) HR features -- so ``fusion=None`` is
    passed to the :class:`FeatureFusionModel` base. Each sample's task logits
    are its ground-truth (train) or predicted (eval, if
    ``pred_domain_for_d3g``) domain's head output. A consistency/relation loss
    (``rel_logits``) additionally blends *other* domains' head outputs, weighted
    by :class:`~models.components.domain_relations.D3GRelation` scores between
    the LR branch features and each candidate domain -- encouraging nearby
    domains' heads to agree. See ``D3GRelation`` for the relation-score formula
    and :meth:`domain_weights` for how it is applied across all heads at once.
    """

    def __init__(
        self,
        branches: DualBranch,
        num_task_labels: int,
        num_domain_labels: int,
        lr_domain_loss_coeff: float = 0.1176,
        consistency_loss_coeff: float = 0.2941,
        learnable_relation_coeff: float = 0.8,
        pred_domain_for_d3g: bool = True,
        detach_lr_for_consistency: bool = False,
        detach_hr_for_consistency: bool = False,
    ):
        """Build the branches, per-domain task heads, and the D3G relation module.

        Args:
            branches (DualBranch): Runs the HR and LR encoders.
            num_task_labels (int): Number of task classes (output width of each
                per-domain head).
            num_domain_labels (int): Number of domains; also the number of
                per-domain task heads (:attr:`num_heads`) and the size of the
                domain classifiers.
            lr_domain_loss_coeff (float): Weight of the LR domain loss in the
                combined training objective (read by
                ``MultiScaleClassificationModule``).
            consistency_loss_coeff (float): Weight of the D3G consistency loss
                (computed from ``rel_logits``) in the combined training
                objective (read by ``MultiScaleClassificationModule``).
            learnable_relation_coeff (float): Passed to
                :class:`~models.components.domain_relations.D3GRelation`;
                blends the fixed domain-indicator relation with a learned
                cosine-similarity relation.
            pred_domain_for_d3g (bool): At eval time, if True, use the LR domain
                classifier's argmax prediction as ``region_ids`` when
                ``region_ids`` is not supplied; if False, require ground-truth
                ``region_ids``.
            detach_lr_for_consistency (bool): If True, detach the LR features
                before computing the domain relation weights used in the
                consistency loss.
            detach_hr_for_consistency (bool): If True, detach the HR features
                before recomputing per-domain head outputs for the consistency
                loss (a second forward pass through ``task_classifiers``,
                separate from the one used for ``task_logits``).
        """
        super().__init__(
            branches=branches,
            fusion=None,
            num_task_labels=num_task_labels,
            num_domain_labels=num_domain_labels,
            lr_domain_loss_coeff=lr_domain_loss_coeff,
        )
        self.detach_lr_for_consistency = detach_lr_for_consistency
        self.detach_hr_for_consistency = detach_hr_for_consistency
        self.num_heads = num_domain_labels
        self.d3g_relation = D3GRelation(learnable_relation_coeff=learnable_relation_coeff, internal_dim=256, lr_features_dim=branches.lr_encoder.out_dim, num_domains=num_domain_labels)
        self.consistency_loss_coeff = consistency_loss_coeff
        self.pred_domain_for_d3g = pred_domain_for_d3g
        self.task_classifier = None
        self.task_classifiers = nn.ModuleList(
            [nn.Linear(branches.hr_encoder.out_dim, num_task_labels) for _ in range(self.num_heads)]
        )

    def forward(self, x: Dict[str, torch.Tensor], region_ids: Optional[torch.Tensor] = None):
        """Route each sample to its domain's task head and compute the D3G consistency signal.

        Args:
            x (Dict[str, torch.Tensor]): Batch dict forwarded to
                :attr:`branches`.
            region_ids (Optional[torch.Tensor]): Shape ``(batch_size,)``,
                integer -- ground-truth domain ids. Required during training.
                During eval, required only if :attr:`pred_domain_for_d3g` is
                False; otherwise ignored in favor of the LR domain classifier's
                prediction.

        Returns:
            Dict[str, torch.Tensor]: ``"task_logits"`` and ``"rel_logits"``
            (each ``(batch_size, num_task_labels)``), plus
            ``"lr_domain_logits"`` / ``"hr_domain_logits"``
            (each ``(batch_size, num_domain_labels)``). In training,
            ``task_logits`` is each sample's own-domain head output and
            ``rel_logits`` is the *other*-domain-weighted average of head
            outputs (the consistency target); in eval, both equal the
            domain-relation-weighted average across all heads.

        Raises:
            ValueError: If ``region_ids`` is ``None`` during training, or during
                eval when :attr:`pred_domain_for_d3g` is False.
        """
        hr_features, lr_features = self.branches(x)

        if self.training:
            if region_ids is None:
                raise ValueError("D3GModel requires region_ids for training.")
        else:  # evaluation mode - here we have the option to use ground truth region ids or predicted ones
            if not self.pred_domain_for_d3g:
                if region_ids is None:
                    raise ValueError("D3GModel requires region_ids for inference when pred_domain_for_d3g is False.")
            else:
                region_ids = self.lr_domain_classifier(lr_features).argmax(dim=1)

        head_outputs = torch.stack(
            [head(hr_features) for head in self.task_classifiers],
            dim=1,
        )

        if self.training:
            task_logits = head_outputs[torch.arange(len(region_ids), device=region_ids.device), region_ids]
            hr_for_consistency = hr_features.detach() if self.detach_hr_for_consistency else hr_features
            head_outputs_detached = torch.stack(
                [head(hr_for_consistency) for head in self.task_classifiers],
                dim=1,
            )
            lr_for_consistency = lr_features.detach() if self.detach_lr_for_consistency else lr_features
            domain_weights = self.domain_weights(region_ids, lr_for_consistency)
            consistency_weights = domain_weights.clone()
            consistency_weights[
                torch.arange(len(region_ids), device=region_ids.device),
                region_ids,
            ] = 0.0
            consistency_weights_sum = torch.clamp(consistency_weights.sum(dim=1), min=1e-6)
            rel_logits = torch.sum(consistency_weights * head_outputs_detached, dim=1) / consistency_weights_sum
        else:
            domain_weights = self.domain_weights(region_ids, lr_features)
            weights_sum = torch.clamp(domain_weights.sum(dim=1), min=1e-6)
            rel_logits = torch.sum(domain_weights * head_outputs, dim=1) / weights_sum
            task_logits = rel_logits

        outputs = {
            "task_logits": task_logits,
            "rel_logits": rel_logits,
            "lr_domain_logits": self.lr_domain_classifier(lr_features),
            "hr_domain_logits": self.hr_domain_classifier(hr_features.detach()),
        }
        return outputs

    def task_parameters(self) -> List[torch.nn.Parameter]:
        """List[torch.nn.Parameter]: Branches + D3G relation module + all per-domain task heads."""
        return (
            list(self.branches.parameters())
            + list(self.d3g_relation.parameters())
            + list(self.task_classifiers.parameters())
        )

    def domain_weights(
        self,
        region_ids: torch.Tensor,
        lr_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute each sample's D3G relation score against every domain head.

        Calls :attr:`d3g_relation` once per head index (0..:attr:`num_heads` -
        1) and concatenates the per-head ``(batch_size, 1)`` scores.

        Args:
            region_ids (torch.Tensor): Shape ``(batch_size,)``, integer --
                each sample's (ground-truth or predicted) domain id, passed to
                :class:`~models.components.domain_relations.D3GRelation` as
                ``domain_id``.
            lr_features (torch.Tensor): Shape ``(batch_size, lr_features_dim)``,
                float32 -- LR branch features, used by the relation module's
                learned term.

        Returns:
            torch.Tensor: Shape ``(batch_size, num_heads, 1)``, float32 --
            relation score of each sample against each domain head, broadcast-
            ready to weight ``(batch_size, num_heads, num_task_labels)`` head
            outputs.
        """
        weights = torch.cat(
            [
                self.d3g_relation(
                    region_ids,
                    lr_features,
                    torch.tensor(head_idx, device=lr_features.device),
                )
                for head_idx in range(self.num_heads)
            ],
            dim=1,
        )
        return weights.unsqueeze(-1)
