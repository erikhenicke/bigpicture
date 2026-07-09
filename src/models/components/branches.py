"""Single-scale image/location encoders ("branches") and the dual-branch wrappers
that pair an HR branch with an LR/location/domain branch.

The abstract :class:`Branch` wraps a pretrained image backbone (built by a
subclass's ``_get_model``) behind a common ``forward(x) -> (batch, out_dim)``
interface, and knows how to surgically resize the backbone's first conv/patch-
embed layer when ``in_channels`` differs from the pretrained 3 (RGB) -- needed
because the LR branch may take a stacked Landsat tensor with more than 3 bands,
optionally augmented with coordinate/positional-encoding channels. Concrete image
backbones: :class:`DenseNetBranch`, :class:`DeitBranch`, :class:`TimmBranch`,
:class:`SatCLIPImageBranch`, :class:`DINOv3Branch`. Non-image encoders:
:class:`SatCLIPLocationBranch` (continuous lat/lon -> embedding) and
:class:`DomainEmbeddingBranch` (discrete domain id -> embedding).

The ``*DualBranch`` classes each run two encoders side by side and return a
``(hr_features, lr_features)`` tuple: :class:`DualBranch` (two image
``Branch``es, with optional coordinate/spatial-encoding augmentation),
:class:`CoordDualBranch` (image + :class:`SatCLIPLocationBranch`), and
:class:`DomainDualBranch` (image + :class:`DomainEmbeddingBranch`). These are
consumed by ``models.components.fusion_models.FeatureFusionModel`` and its
subclasses.
"""
import os
import sys

import torch
import torch.nn as nn
from abc import abstractmethod
from typing import Dict, Optional

from models.components.spatial_encoding import SpatialEncoding


VALID_EXTRA_CH_INITS = ("average", "zero", "he")


class Branch(nn.Module):
    """Abstract base for a single-scale image encoder wrapping a pretrained backbone.

    Subclasses implement :meth:`_get_model` (build/load the backbone),
    :meth:`_adapt_input_channels` (resize its stem to ``in_channels`` when not 3),
    and :attr:`out_dim` (the backbone's output feature width); ``forward`` (not
    defined here) must return a ``(batch_size, out_dim)`` feature vector. This
    base class handles the shared bookkeeping: validating ``in_channels`` /
    ``landsat_channel_init``, invoking ``_get_model``, and triggering
    ``_adapt_input_channels`` when the requested input width differs from the
    pretrained 3-channel (RGB) stem. :meth:`_reorder_pretrained_weights` and
    :meth:`_init_extra_weights` are shared helpers subclasses use inside their
    ``_adapt_input_channels`` implementation to seed the resized stem's weights.
    """

    def __init__(self, in_channels: int = 3, landsat_channel_init: str = "zero", stacked: bool = False, bgr_input: bool = False, **kwargs):
        """Build the backbone and, if needed, adapt its input stem.

        Args:
            in_channels (int): Number of input channels this branch's stem
                should accept. Must be ``>= 3``; values above 3 trigger
                :meth:`_adapt_input_channels` to resize the pretrained stem.
            landsat_channel_init (str): How channels beyond the pretrained 3 are
                initialized; one of :data:`VALID_EXTRA_CH_INITS`
                (``"average"``, ``"zero"``, ``"he"``). See
                :meth:`_init_extra_weights`.
            stacked (bool): Whether the extra input channels are stacked Landsat
                bands whose first 3 are the visible (B, G, R) bands -- affects
                how ``"average"`` initialization treats those first 3 extra
                channels. See :meth:`_init_extra_weights`.
            bgr_input (bool): Whether this branch's input channel order is BGR
                rather than RGB, so the pretrained RGB stem weights must be
                channel-reordered when copied in. See
                :meth:`_reorder_pretrained_weights`.
            **kwargs: Forwarded to :meth:`_get_model` (e.g. ``pretrained``,
                ``model_name``).

        Raises:
            ValueError: If ``in_channels < 3``, or if ``landsat_channel_init`` is
                not one of :data:`VALID_EXTRA_CH_INITS`.
        """
        super().__init__()

        if in_channels < 3:
            raise ValueError(f"Unsupported number of input channels: {in_channels}. Must be >= 3.")
        if landsat_channel_init not in VALID_EXTRA_CH_INITS:
            raise ValueError(f"Unsupported landsat_channel_init: {landsat_channel_init}. Must be one of {VALID_EXTRA_CH_INITS}.")

        self.in_channels = in_channels
        self.landsat_channel_init = landsat_channel_init
        self.stacked = stacked
        self.bgr_input = bgr_input
        self.model = self._get_model(**kwargs)

        if in_channels != 3:
            self._adapt_input_channels(in_channels)

    @abstractmethod
    def _get_model(self, **kwargs) -> nn.Module:
        """Build (and, if applicable, load pretrained weights for) the backbone.

        Args:
            **kwargs: Subclass-specific construction options (e.g. ``pretrained``,
                ``model_name``), forwarded from :meth:`__init__`.

        Returns:
            nn.Module: The wrapped backbone, assigned to ``self.model``.
        """
        pass

    @abstractmethod
    def _adapt_input_channels(self, in_channels: int) -> None:
        """Resize the backbone's first conv/patch-embed layer to accept ``in_channels``.

        Args:
            in_channels (int): Target number of input channels for the stem.
        """
        pass

    @property
    @abstractmethod
    def out_dim(self) -> int:
        """int: Dimensionality of the feature vector returned by ``forward``."""
        pass

    def _reorder_pretrained_weights(self, weight: torch.Tensor) -> torch.Tensor:
        """Swap a pretrained RGB conv weight's channel order to BGR if needed.

        Args:
            weight (torch.Tensor): Shape ``(out_channels, 3, kH, kW)`` -- the
                pretrained stem's RGB-ordered conv/patch-embed weight.

        Returns:
            torch.Tensor: Same shape as ``weight``. If :attr:`bgr_input` is True,
            channels reordered to BGR (index ``[2, 1, 0]``); otherwise ``weight``
            unchanged.
        """
        if self.bgr_input:
            return weight[:, [2, 1, 0], :, :]
        return weight

    def _init_extra_weights(self, weight_slice: torch.Tensor, old_weight: torch.Tensor) -> None:
        """In-place initialize the resized stem's weights for channels beyond the pretrained 3.

        Behavior depends on :attr:`landsat_channel_init`:

        - ``"average"``: if :attr:`stacked` is True and there are at least 3 extra
          channels, the first 3 (assumed to be Landsat's visible B, G, R bands)
          are seeded from the pretrained RGB weights reordered to BGR, and every
          remaining extra channel (or all extra channels, if not stacked / fewer
          than 3) is seeded with the per-output-channel mean of the pretrained
          weights, broadcast across the remaining input channels.
        - ``"zero"``: remaining extra channels are zeroed.
        - ``"he"``: remaining extra channels use Kaiming-normal init
          (``mode="fan_out"``, ``nonlinearity="relu"``).

        Args:
            weight_slice (torch.Tensor): Shape ``(out_channels, num_extra_channels,
                kH, kW)`` -- the slice of the new stem's weight tensor
                corresponding to the input channels beyond the pretrained 3;
                modified in place.
            old_weight (torch.Tensor): Shape ``(out_channels, 3, kH, kW)`` -- the
                pretrained stem's original (un-reordered, RGB) weight, used as
                the initialization source.

        Returns:
            None: ``weight_slice`` is modified in place.
        """
        with torch.no_grad():
            # "average" reuses the pretrained RGB weights for the stacked Landsat
            # visible bands (B, G, R) and only initialises the remaining bands.
            # "zero"/"he" instead override *every* non-RGB channel, so only the
            # FMoW-RGB stem channels stay pretrained.
            if self.landsat_channel_init == "average" and self.stacked and weight_slice.size(1) >= 3:
                # Landsat visible bands (B, G, R): reuse pretrained RGB swapped to BGR
                weight_slice[:, :3, :, :].copy_(old_weight[:, [2, 1, 0], :, :])
                remaining = weight_slice[:, 3:, :, :]
            else:
                remaining = weight_slice

            if remaining.numel() == 0:
                return

            if self.landsat_channel_init == "average":
                mean_weight = old_weight.mean(dim=1, keepdim=True)
                remaining.copy_(mean_weight.repeat(1, remaining.size(1), 1, 1))
            elif self.landsat_channel_init == "zero":
                remaining.zero_()
            elif self.landsat_channel_init == "he":
                nn.init.kaiming_normal_(remaining, mode="fan_out", nonlinearity="relu")


class DenseNetBranch(Branch):
    """DenseNet-121 image encoder (torchvision), optionally ImageNet-pretrained.

    Feeds the input through DenseNet's conv feature extractor, then a ReLU and a
    global average pool, producing a single feature vector per image.
    """

    def __init__(self, in_channels: int = 3, pretrained: bool = True, landsat_channel_init: str = "zero", stacked: bool = False, bgr_input: bool = False):
        """Build a DenseNet-121 backbone and adapt its stem if needed.

        Args:
            in_channels (int): Number of input channels; see :class:`Branch`.
            pretrained (bool): If True, load ImageNet-1K pretrained weights.
            landsat_channel_init (str): See :class:`Branch`.
            stacked (bool): See :class:`Branch`.
            bgr_input (bool): See :class:`Branch`.
        """
        super().__init__(in_channels=in_channels, landsat_channel_init=landsat_channel_init, stacked=stacked, bgr_input=bgr_input, pretrained=pretrained)
    
    def forward(self, x):
        """Encode a batch of images to feature vectors.

        Args:
            x (torch.Tensor): Shape ``(batch_size, in_channels, H, W)``, float32.

        Returns:
            torch.Tensor: Shape ``(batch_size, out_dim)``, float32 -- globally
            average-pooled DenseNet feature map.
        """
        features = self.model.features(x)
        out = nn.functional.relu(features, inplace=False)
        # Adaptive average pooling sets kernel size and stride automatically, with (1, 1) global average pooling is applied to each feature map.
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        return torch.flatten(out, 1)

    @property
    def out_dim(self) -> int:
        """int: DenseNet-121's classifier input width (1024)."""
        return self.model.classifier.in_features

    def _get_model(self, pretrained=True):
        """Build torchvision's ``densenet121``.

        Args:
            pretrained (bool): If True, load ``IMAGENET1K_V1`` weights.

        Returns:
            nn.Module: A torchvision ``DenseNet`` instance.
        """
        from torchvision.models import densenet121
        weights = 'IMAGENET1K_V1' if pretrained else None
        return densenet121(weights=weights)

    def _adapt_input_channels(self, in_channels):
        """Replace ``model.features.conv0`` with a version accepting ``in_channels``.

        No-op if the existing stem already accepts ``in_channels``. Otherwise
        builds a new conv with the same kernel/stride/padding/bias, copies the
        (possibly channel-reordered) pretrained weights into the first 3 (or
        ``old_conv.in_channels``) input channels, and initializes any additional
        channels via :meth:`Branch._init_extra_weights`.

        Args:
            in_channels (int): Target number of input channels for the stem.
        """
        old_conv = self.model.features.conv0

        if old_conv.in_channels == in_channels:
            return

        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        with torch.no_grad():
            new_conv.weight[:, : old_conv.in_channels, :, :] = self._reorder_pretrained_weights(old_conv.weight)
            if in_channels > old_conv.in_channels:
                self._init_extra_weights(
                    new_conv.weight[:, old_conv.in_channels:, :, :],
                    old_conv.weight,
                )
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        self.model.features.conv0 = new_conv

class DeitBranch(Branch):
    """DeiT-tiny ViT image encoder (HuggingFace ``transformers``).

    Wraps ``ViTForImageClassification`` (DeiT-tiny/16, 224x224) and returns the
    CLS token embedding as the feature vector.
    """

    def __init__(self, in_channels: int = 3, pretrained: bool = True, landsat_channel_init: str = "zero", stacked: bool = False, bgr_input: bool = False):
        """Build a DeiT-tiny backbone and adapt its patch-embedding stem if needed.

        Args:
            in_channels (int): Number of input channels; see :class:`Branch`.
            pretrained (bool): If True, load ``facebook/deit-tiny-patch16-224``
                pretrained weights; otherwise build an untrained model with that
                config, sized for ``in_channels`` input channels directly.
            landsat_channel_init (str): See :class:`Branch`.
            stacked (bool): See :class:`Branch`.
            bgr_input (bool): See :class:`Branch`.
        """
        super().__init__(in_channels=in_channels, landsat_channel_init=landsat_channel_init, stacked=stacked, bgr_input=bgr_input, pretrained=pretrained)

    def forward(self, x):
        """Encode a batch of images to their CLS token embedding.

        Args:
            x (torch.Tensor): Shape ``(batch_size, in_channels, H, W)``, float32.

        Returns:
            torch.Tensor: Shape ``(batch_size, out_dim)``, float32 -- the CLS
            token (position 0) of the ViT's last hidden state.
        """
        tokens = self.model.vit(x).last_hidden_state
        return tokens[:, 0, :]

    @property
    def out_dim(self) -> int:
        """int: The ViT's hidden size."""
        return self.model.config.hidden_size

    def _get_model(self, pretrained=True):
        """Build ``ViTForImageClassification`` from the DeiT-tiny config.

        Args:
            pretrained (bool): If True, load pretrained weights from
                ``facebook/deit-tiny-patch16-224``. If False, build an untrained
                model from that config with ``num_channels=self.in_channels``
                (so no separate stem-adaptation copy is needed for the untrained
                path; :meth:`_adapt_input_channels` still runs afterward but is a
                cheap re-projection since there is no pretrained weight to copy).

        Returns:
            nn.Module: A ``ViTForImageClassification`` instance with
            ``output_hidden_states=True``.
        """
        from transformers import ViTForImageClassification, ViTConfig

        if pretrained:
            return ViTForImageClassification.from_pretrained(
                'facebook/deit-tiny-patch16-224',
                output_hidden_states=True,
                ignore_mismatched_sizes=True,
                use_safetensors=True,
            )
        return ViTForImageClassification(ViTConfig.from_pretrained(
            'facebook/deit-tiny-patch16-224',
            output_hidden_states=True,
            num_labels=1000,
            num_channels=self.in_channels,
        ))


    def _adapt_input_channels(self, in_channels: int) -> None:
        """Replace the ViT patch-embedding projection conv to accept ``in_channels``.

        Builds a new conv with the same kernel/stride/padding/bias as the
        existing patch-embedding projection, copies the (possibly
        channel-reordered) pretrained weights into the first
        ``old_projection.in_channels`` input channels, initializes any
        additional channels via :meth:`Branch._init_extra_weights`, and updates
        the patch-embedding module's and model config's ``num_channels``.

        Args:
            in_channels (int): Target number of input channels for the stem.
        """
        patch_embeddings = self.model.vit.embeddings.patch_embeddings
        old_projection = patch_embeddings.projection

        new_projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_projection.out_channels,
            kernel_size=old_projection.kernel_size,
            stride=old_projection.stride,
            padding=old_projection.padding,
            bias=old_projection.bias is not None,
        )

        with torch.no_grad():
            new_projection.weight[:, :old_projection.in_channels, :, :] = self._reorder_pretrained_weights(old_projection.weight)
            if in_channels > old_projection.in_channels:
                self._init_extra_weights(
                    new_projection.weight[:, old_projection.in_channels:, :, :],
                    old_projection.weight,
                )
            if old_projection.bias is not None:
                new_projection.bias.copy_(old_projection.bias)

        patch_embeddings.projection = new_projection
        patch_embeddings.num_channels = in_channels
        self.model.config.num_channels = in_channels
        self.model.vit.config.num_channels = in_channels


class TimmBranch(Branch):
    """Generic branch backed by a ``timm`` model.

    ``timm.create_model`` handles multi-channel stems natively via ``in_chans``,
    so this branch does not need to surgically rewrite the first conv like
    ``DenseNetBranch`` / ``DeitBranch`` do. Works for both ``efficientformerv2_s1``
    and ``tf_efficientnetv2_b1`` (and most other timm backbones).
    """

    def __init__(self, model_name: str, in_channels: int = 3, pretrained: bool = True, landsat_channel_init: str = "zero", stacked: bool = False, bgr_input: bool = False):
        """Build a ``timm`` backbone sized for ``in_channels`` directly.

        Args:
            model_name (str): ``timm`` model identifier (e.g.
                ``"efficientformerv2_s1"``, ``"tf_efficientnetv2_b1"``).
            in_channels (int): Number of input channels; see :class:`Branch`.
                Passed straight to ``timm.create_model(in_chans=...)``, so
                ``landsat_channel_init`` / ``stacked`` / ``bgr_input`` are stored
                on the instance but unused (``_adapt_input_channels`` is a no-op).
            pretrained (bool): If True, load ``timm``'s pretrained weights.
            landsat_channel_init (str): Stored but unused; see class docstring.
            stacked (bool): Stored but unused; see class docstring.
            bgr_input (bool): Stored but unused; see class docstring.
        """
        super().__init__(in_channels=in_channels, landsat_channel_init=landsat_channel_init, stacked=stacked, bgr_input=bgr_input, model_name=model_name, pretrained=pretrained)

    def forward(self, x):
        """Encode a batch of images to feature vectors.

        Args:
            x (torch.Tensor): Shape ``(batch_size, in_channels, H, W)``, float32.

        Returns:
            torch.Tensor: Shape ``(batch_size, out_dim)``, float32 -- the
            ``timm`` model's globally-pooled features (``num_classes=0``,
            ``global_pool="avg"``).
        """
        return self.model(x)

    @property
    def out_dim(self) -> int:
        """int: The ``timm`` backbone's feature width (``model.num_features``)."""
        return self.model.num_features

    def _get_model(self, model_name: str, pretrained: bool = True) -> nn.Module:
        """Build the ``timm`` model with a stem sized for ``self.in_channels``.

        Args:
            model_name (str): ``timm`` model identifier.
            pretrained (bool): If True, load pretrained weights.

        Returns:
            nn.Module: A ``timm`` model with ``num_classes=0`` and
            ``global_pool="avg"`` (i.e. it outputs pooled features, not logits).
        """
        import timm

        return timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=self.in_channels,
            num_classes=0,
            global_pool="avg",
        )

    def _adapt_input_channels(self, in_channels: int) -> None:
        """No-op: ``timm.create_model`` already built the stem for ``in_channels``.

        Args:
            in_channels (int): Unused; present to satisfy the :class:`Branch`
                interface.
        """
        # timm has already built the stem with the requested in_chans and
        # copied pretrained weights (averaging / repeating as needed), so the
        # base-class hook has nothing left to do.
        return


class DualBranch(nn.Module):
    """Runs an HR image encoder and an LR (Landsat) image encoder side by side.

    Optionally augments each branch's input with raw normalized coordinate
    channels and/or a Fourier positional encoding (both computed from a
    per-pixel coordinate grid), and always appends an overlap mask to the LR
    input when one is present in the batch. The HR/LR encoders must already be
    built with matching ``in_channels`` (3 + any enabled HR extras, and
    ``landsat_channels`` + any enabled LR extras) -- see ``make_model`` in
    ``train/run_experiment.py``, which computes those extra-channel counts and
    instantiates the encoders accordingly before building this wrapper.
    """

    def __init__(
        self,
        hr_encoder: Branch,
        lr_encoder: Branch,
        landsat_channels: int = 6,
        coord_channels_hr: bool = False,
        coord_channels_lr: bool = False,
        hr_spatial_encoding: Optional[SpatialEncoding] = None,
        lr_spatial_encoding: Optional[SpatialEncoding] = None,
    ):
        """Store the two encoders and the input-augmentation configuration.

        Args:
            hr_encoder (Branch): Encoder applied to the (possibly augmented) HR
                RGB image.
            lr_encoder (Branch): Encoder applied to the (possibly augmented) LR
                Landsat image.
            landsat_channels (int): Number of leading channels to keep from the
                raw ``x["landsat"]`` tensor before any augmentation.
            coord_channels_hr (bool): If True, concatenate the raw
                ``x["coord_grid_hr"]`` coordinate grid onto the HR image.
            coord_channels_lr (bool): If True, concatenate the raw
                ``x["coord_grid_lr"]`` coordinate grid onto the LR image.
            hr_spatial_encoding (Optional[SpatialEncoding]): If given (and
                ``x["coord_grid_hr"]`` is present), applied to the HR coordinate
                grid and the result concatenated onto the HR image.
            lr_spatial_encoding (Optional[SpatialEncoding]): If given (and
                ``x["coord_grid_lr"]`` is present), applied to the LR coordinate
                grid and the result concatenated onto the LR image.
        """
        super().__init__()

        self.hr_encoder = hr_encoder
        self.lr_encoder = lr_encoder
        self.landsat_channels = landsat_channels
        self.coord_channels_hr = coord_channels_hr
        self.coord_channels_lr = coord_channels_lr
        self.hr_spatial_encoding = hr_spatial_encoding
        self.lr_spatial_encoding = lr_spatial_encoding

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Assemble each branch's (possibly augmented) input and encode it.

        Args:
            x (Dict[str, torch.Tensor]): Batch dict with:

                - ``"rgb"``: ``(batch_size, 3, H_hr, W_hr)``, float32 -- HR image.
                - ``"landsat"``: ``(batch_size, C, H_lr, W_lr)``, float32 -- raw
                  Landsat image; only the first :attr:`landsat_channels` channels
                  are used.
                - ``"coord_grid_hr"`` (optional): ``(batch_size, 2, H_hr, W_hr)``,
                  float32 -- required if :attr:`coord_channels_hr` is True or
                  :attr:`hr_spatial_encoding` is set.
                - ``"coord_grid_lr"`` (optional): ``(batch_size, 2, H_lr, W_lr)``,
                  float32 -- required if :attr:`coord_channels_lr` is True or
                  :attr:`lr_spatial_encoding` is set.
                - ``"overlap_mask"`` (optional): ``(batch_size, 1, H_lr, W_lr)``,
                  float32 -- if present, always concatenated onto the LR image.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: ``(hr_features, lr_features)``,
            shapes ``(batch_size, hr_encoder.out_dim)`` and
            ``(batch_size, lr_encoder.out_dim)`` respectively, both float32.
            (Annotated as a single ``torch.Tensor`` in the signature, but a
            2-tuple is returned.)
        """
        hr_image = x["rgb"]
        lr_image = x["landsat"][:, : self.landsat_channels, :, :]

        extra_hr, extra_lr = [], []

        if self.coord_channels_hr:
            extra_hr.append(x["coord_grid_hr"])
        if self.coord_channels_lr:
            extra_lr.append(x["coord_grid_lr"])

        if "overlap_mask" in x:
            extra_lr.append(x["overlap_mask"])

        if self.hr_spatial_encoding is not None and "coord_grid_hr" in x:
            extra_hr.append(self.hr_spatial_encoding(x["coord_grid_hr"]))
        if self.lr_spatial_encoding is not None and "coord_grid_lr" in x:
            extra_lr.append(self.lr_spatial_encoding(x["coord_grid_lr"]))

        if extra_hr:
            hr_image = torch.cat([hr_image] + extra_hr, dim=1)
        if extra_lr:
            lr_image = torch.cat([lr_image] + extra_lr, dim=1)

        hr_features = self.hr_encoder(hr_image)
        lr_features = self.lr_encoder(lr_image)
        return hr_features, lr_features


class SatCLIPImageBranch(Branch):
    """Pretrained ViT branch initialized with SatCLIP weights.
    
    > Image encoder: We train SatCLIP models with ViT16, ResNet18 and ResNet50 image encoders, all pretrained on Sentinel-2 imagery and published by Wang et al. (2022b). We keep the image encoders frozen during training, and only train a last projection layer that maps the image embeddings into the desired output space. We find this to be ideal for training at a size of 256–this is equivalent to the embedding size used by CSP Mai et al. (2023).

    See lib/satclip/satclip/model.py:304 for details on the SatCLIP vision model initialisation.
    """

    # Sentinel-2 band order (torchgeo): B1,B2,B3,B4,B5,B6,B7,B8,B8a,B9,B10,B11,B12
    # RGB = B4(Red) idx 3, B3(Green) idx 2, B2(Blue) idx 1
    S2_RGB_INDICES = [3, 2, 1]

    def __init__(
        self,
        in_channels: int = 3,
        pretrained: bool = True,
        landsat_channel_init: str = "zero",
        stacked: bool = False,
        bgr_input: bool = False,
        unfreeze_all: bool = False,
        unfreeze_first_block: bool = False,
        unfreeze_head: bool = False,
        num_unfreeze_last: int = 0,
    ):
        """Build the SatCLIP ViT backbone and apply the requested freezing scheme.

        Freezing only applies when ``pretrained`` is True; an untrained model is
        left fully trainable regardless of the other flags.

        Args:
            in_channels (int): Number of input channels; see :class:`Branch`.
            pretrained (bool): If True, load SatCLIP's pretrained ViT-S/16
                weights (Sentinel-2, via ``microsoft/SatCLIP-ViT16-L10``).
            landsat_channel_init (str): See :class:`Branch`.
            stacked (bool): See :class:`Branch`.
            bgr_input (bool): See :class:`Branch`.
            unfreeze_all (bool): If True (and ``pretrained``), leave the entire
                backbone trainable, ignoring the other unfreeze flags.
            unfreeze_first_block (bool): If True (and ``pretrained`` and not
                ``unfreeze_all``), also unfreeze the first transformer block.
                See :meth:`_freeze`.
            unfreeze_head (bool): If True (and ``pretrained`` and not
                ``unfreeze_all``), also unfreeze the classification head. See
                :meth:`_freeze`.
            num_unfreeze_last (int): If ``> 0`` (and ``pretrained`` and not
                ``unfreeze_all``), also unfreeze the final norm layer and this
                many trailing transformer blocks. See :meth:`_freeze`.
        """
        super().__init__(in_channels=in_channels, landsat_channel_init=landsat_channel_init, stacked=stacked, bgr_input=bgr_input, pretrained=pretrained)
        if pretrained:
            if unfreeze_all:
                self.model.requires_grad_(True)
            else:
                self._freeze(unfreeze_first_block, unfreeze_head, num_unfreeze_last)

    def _freeze(self, unfreeze_first_block: bool, unfreeze_head: bool, num_unfreeze_last: int) -> None:
        """Freeze the backbone except the patch-embed projection and any selected parts.

        Always leaves ``patch_embed.proj`` trainable (since it may have been
        resized for a non-3-channel input); selectively unfreezes the first
        block, the head, and/or the final norm + last ``num_unfreeze_last``
        blocks, per the flags.

        Args:
            unfreeze_first_block (bool): If True, unfreeze ``model.blocks[0]``.
            unfreeze_head (bool): If True, unfreeze ``model.head``.
            num_unfreeze_last (int): If ``> 0``, unfreeze ``model.norm`` and the
                last ``num_unfreeze_last`` entries of ``model.blocks``.
        """
        self.model.requires_grad_(False)

        self.model.patch_embed.proj.requires_grad_(True)

        blocks = self.model.blocks
        if unfreeze_first_block:
            blocks[0].requires_grad_(True)
        if unfreeze_head: 
            self.model.head.requires_grad_(True)
        if num_unfreeze_last > 0:
            self.model.norm.requires_grad_(True)
            for block in blocks[-num_unfreeze_last:]:
                block.requires_grad_(True)

    def forward(self, x):
        """Encode a batch of images with the SatCLIP ViT and its projection head.

        Args:
            x (torch.Tensor): Shape ``(batch_size, in_channels, H, W)``, float32.

        Returns:
            torch.Tensor: Shape ``(batch_size, out_dim)``, float32 -- the ViT's
            projection-head output (``embed_dim`` from the SatCLIP checkpoint).
        """
        return self.model(x)

    @property
    def out_dim(self) -> int:
        """int: The SatCLIP projection head's output width (``embed_dim``)."""
        return self.model.head.out_features

    def _get_model(self, pretrained=True):
        """Build a ``vit_small_patch16_224`` and load SatCLIP's visual-encoder weights.

        Downloads the ``microsoft/SatCLIP-ViT16-L10`` checkpoint, extracts the
        ``model.visual.*`` state dict entries (the image encoder), remaps the
        patch-embedding projection's input channels from Sentinel-2's 13 bands
        down to the 3 RGB-equivalent bands (:data:`S2_RGB_INDICES`), and loads
        the result into a freshly created ``timm`` ViT.

        Args:
            pretrained (bool): Currently always treated as pretrained -- this
                method always downloads and loads the SatCLIP checkpoint
                regardless of the argument value.

        Returns:
            nn.Module: A ``timm`` ``vit_small_patch16_224`` with SatCLIP's
            pretrained visual-encoder weights (3-channel RGB stem,
            ``num_classes=hp["embed_dim"]`` projection head).
        """
        import timm
        from huggingface_hub import hf_hub_download

        ckpt_path = hf_hub_download(
            "microsoft/SatCLIP-ViT16-L10",
            filename="satclip-vit16-l10.ckpt",
        )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hp = ckpt["hyper_parameters"]

        model = timm.create_model(
            "vit_small_patch16_224",
            in_chans=3,
            num_classes=hp["embed_dim"],
        )

        state_dict = {
            k.removeprefix("model.visual."): v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("model.visual.")
        }
        state_dict["patch_embed.proj.weight"] = (
            state_dict["patch_embed.proj.weight"][:, self.S2_RGB_INDICES, :, :]
        )
        model.load_state_dict(state_dict)

        return model

    def _adapt_input_channels(self, in_channels):
        """No-op: this branch always uses a fixed 3-channel (RGB) stem.

        Args:
            in_channels (int): Unused; present to satisfy the :class:`Branch`
                interface. ``in_channels != 3`` is silently accepted without
                actually resizing the stem.
        """
        pass


class DINOv3Branch(Branch):
    """DINOv3 ViT image encoder (HuggingFace ``transformers``), base or large.

    Wraps a pretrained ``facebook/dinov3-vit{b,l}16-pretrain-*`` model and
    returns its pooled output as the feature vector.
    """

    HF_MODEL_LARGE = "facebook/dinov3-vitl16-pretrain-sat493m"
    HF_MODEL_BASE = "facebook/dinov3-vitb16-pretrain-lvd1689m"

    def __init__(
        self,
        in_channels: int = 3,
        pretrained: bool = True,
        landsat_channel_init: str = "zero",
        stacked: bool = False,
        bgr_input: bool = False,
        model_size: str = "base",
        freeze: bool = False,
    ):
        """Build the DINOv3 backbone and adapt/freeze it as requested.

        Args:
            in_channels (int): Number of input channels; see :class:`Branch`.
            pretrained (bool): If True, load a pretrained DINOv3 checkpoint (see
                :meth:`_get_model`).
            landsat_channel_init (str): See :class:`Branch`.
            stacked (bool): See :class:`Branch`.
            bgr_input (bool): See :class:`Branch`.
            model_size (str): ``"base"`` selects :data:`HF_MODEL_BASE`; anything
                else selects :data:`HF_MODEL_LARGE` (only when ``pretrained``).
            freeze (bool): If True, freeze every backbone parameter after
                construction (including any resized patch-embedding stem).
        """
        super().__init__(in_channels=in_channels, landsat_channel_init=landsat_channel_init, stacked=stacked, bgr_input=bgr_input, pretrained=pretrained, model_size=model_size)

        if freeze:
            self.model.requires_grad_(False)

    def forward(self, x):
        """Encode a batch of images with DINOv3 and return the pooled output.

        Args:
            x (torch.Tensor): Shape ``(batch_size, in_channels, H, W)``, float32.

        Returns:
            torch.Tensor: Shape ``(batch_size, out_dim)``, float32 -- the
            model's ``pooler_output``.
        """
        return self.model(x).pooler_output

    @property
    def out_dim(self) -> int:
        """int: The DINOv3 model's hidden size."""
        return self.model.config.hidden_size

    def _get_model(self, pretrained: bool = True, model_size: str = "base") -> nn.Module:
        """Load a pretrained DINOv3 checkpoint, or build an untrained model from its config.

        Args:
            pretrained (bool): If True, load :data:`HF_MODEL_BASE` or
                :data:`HF_MODEL_LARGE` (per ``model_size``) via
                ``AutoModel.from_pretrained``.
            model_size (str): ``"base"`` or ``"large"``; selects which
                checkpoint/config to use.

        Returns:
            nn.Module: A ``transformers`` ``AutoModel`` instance.

        Raises:
            AttributeError: If ``pretrained`` is False -- the untrained path
                reads ``self.HF_MODEL_NAME``, which is not defined on this class
                (only :data:`HF_MODEL_BASE` / :data:`HF_MODEL_LARGE` exist).
                TODO: this looks like a latent bug (untrained DINOv3Branch
                construction is broken); left as-is since fixing logic is out of
                scope for a docstring pass.
        """
        from transformers import AutoModel, AutoConfig

        if pretrained:
            if model_size == "base":
                return AutoModel.from_pretrained(self.HF_MODEL_BASE)
            else:
                return AutoModel.from_pretrained(self.HF_MODEL_LARGE)

        config = AutoConfig.from_pretrained(self.HF_MODEL_NAME)
        return AutoModel.from_config(config)

    def _adapt_input_channels(self, in_channels: int) -> None:
        """Replace the patch-embedding projection conv to accept ``in_channels``.

        No-op if the existing stem already accepts ``in_channels``. Otherwise
        builds a new conv with the same kernel/stride/padding/bias, copies the
        (possibly channel-reordered) pretrained weights into the first
        ``old_proj.in_channels`` input channels, and initializes any additional
        channels via :meth:`Branch._init_extra_weights`.

        Args:
            in_channels (int): Target number of input channels for the stem.
        """
        old_proj = self.model.embeddings.patch_embeddings

        if old_proj.in_channels == in_channels:
            return

        new_proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
            bias=old_proj.bias is not None,
        )

        with torch.no_grad():
            new_proj.weight[:, :old_proj.in_channels, :, :] = self._reorder_pretrained_weights(old_proj.weight)
            if in_channels > old_proj.in_channels:
                self._init_extra_weights(
                    new_proj.weight[:, old_proj.in_channels:, :, :],
                    old_proj.weight,
                )
            if old_proj.bias is not None:
                new_proj.bias.copy_(old_proj.bias)

        self.model.embeddings.patch_embeddings = new_proj


class SatCLIPLocationBranch(nn.Module):
    """Continuous location encoder: (lon, lat) -> embedding, via SatCLIP's encoder.

    Wraps ``lib/satclip/satclip/location_encoder.LocationEncoder`` with a
    spherical-harmonics positional encoding and a SIREN neural network, run in
    double precision internally (spherical harmonics are numerically sensitive)
    and cast back to float32 on output.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        legendre_polys: int = 10,
        capacity: int = 512,
        num_hidden_layers: int = 2,
    ):
        """Build the spherical-harmonics positional encoding and SIREN network.

        Args:
            embed_dim (int): Output embedding dimension (also exposed via
                :attr:`out_dim`).
            legendre_polys (int): Number of Legendre polynomials used by the
                spherical-harmonics positional encoding.
            capacity (int): Hidden layer width of the SIREN network.
            num_hidden_layers (int): Number of hidden layers in the SIREN
                network.
        """
        super().__init__()

        sys.path.append(os.path.join(os.getcwd(), "lib/satclip/satclip"))
        from location_encoder import get_positional_encoding, get_neural_network, LocationEncoder

        posenc = get_positional_encoding("sphericalharmonics", legendre_polys=legendre_polys)
        nnet = get_neural_network(
            "siren",
            input_dim=posenc.embedding_dim,
            num_classes=embed_dim,
            dim_hidden=capacity,
            num_layers=num_hidden_layers,
        )
        self.location_encoder = LocationEncoder(posenc, nnet).double()
        self._embed_dim = embed_dim

    @property
    def out_dim(self) -> int:
        """int: The location embedding dimension (``embed_dim``)."""
        return self._embed_dim

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Encode a batch of coordinates to location embeddings.

        Args:
            coords (torch.Tensor): Shape ``(batch_size, 2)``. As supplied by
                ``FMoWMultiScaleDataset`` (``x["coords"]``), this is
                ``(lat, lon)`` per sample (columns taken from the metadata's
                ``"lat"``, ``"lon"`` fields, in that order).
                TODO: the underlying ``SphericalHarmonics.forward`` (in
                ``lib/satclip/satclip/positional_encoding/spherical_harmonics.py``)
                unpacks its input as ``lon, lat = lonlat[:, 0], lonlat[:, 1]``,
                i.e. it expects ``(lon, lat)`` order -- the opposite of what the
                dataset provides. This looks like a possible pre-existing
                ordering bug rather than intentional; left unverified/unfixed
                here since fixing logic is out of scope for a docstring pass.

        Returns:
            torch.Tensor: Shape ``(batch_size, embed_dim)``, float32 -- the
            location embedding (computed in float64 internally, cast back to
            float32).
        """
        embeddings = self.location_encoder(coords.double())
        return embeddings.float()


class CoordDualBranch(nn.Module):
    """Pairs an HR image encoder with a :class:`SatCLIPLocationBranch`.

    Conditions on continuous geographic coordinates (``x["coords"]``) instead of
    a second image modality; mirrored by :class:`DomainDualBranch`, which
    conditions on a discrete domain label instead.
    """

    def __init__(self, hr_encoder: Branch, lr_encoder: SatCLIPLocationBranch):
        """Store the HR image encoder and the location encoder.

        Args:
            hr_encoder (Branch): Encoder applied to ``x["rgb"]``.
            lr_encoder (SatCLIPLocationBranch): Encoder applied to ``x["coords"]``.
        """
        super().__init__()
        self.hr_encoder = hr_encoder
        self.lr_encoder = lr_encoder

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode the HR image and the geographic coordinates independently.

        Args:
            x (Dict[str, torch.Tensor]): Batch dict with ``"rgb"``
                (``(batch_size, 3, H, W)``, float32) and ``"coords"``
                (``(batch_size, 2)``, see :meth:`SatCLIPLocationBranch.forward`).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: ``(hr_features, lr_features)``,
            shapes ``(batch_size, hr_encoder.out_dim)`` and
            ``(batch_size, lr_encoder.out_dim)`` respectively, both float32.
            (Annotated as a single ``torch.Tensor`` in the signature, but a
            2-tuple is returned.)
        """
        hr_features = self.hr_encoder(x["rgb"])
        lr_features = self.lr_encoder(x["coords"])
        return hr_features, lr_features


class DomainEmbeddingBranch(nn.Module):
    """Location-encoder ablation from Crasto & Rolf (2026), Sec. 5.3.

    Replaces the continuous location encoder with a domain encoder that maps each
    discrete domain label to a learnable embedding used for conditioning. The
    auxiliary domain-prediction loss is applied to these embeddings downstream (via
    ``lr_domain_classifier``) to keep the per-domain vectors separated and avoid
    collapse. Output dimension is matched to the location encoder it ablates against
    so the fusion module is identical.

    This requires the domain label as input at inference
    time as well as during training.
    """

    def __init__(self, num_domains: int = 6, embed_dim: int = 512):
        """Build the domain-id embedding table.

        Args:
            num_domains (int): Number of discrete domains (size of the
                embedding table).
            embed_dim (int): Embedding dimension (also exposed via
                :attr:`out_dim`); matched to the location encoder this ablates.
        """
        super().__init__()
        self.embedding = nn.Embedding(num_domains, embed_dim)
        self._embed_dim = embed_dim

    @property
    def out_dim(self) -> int:
        """int: The domain embedding dimension (``embed_dim``)."""
        return self._embed_dim

    def forward(self, domain_ids: torch.Tensor) -> torch.Tensor:
        """Look up each sample's domain embedding.

        Args:
            domain_ids (torch.Tensor): Shape ``(batch_size,)``, integer (or
                integer-valued) dtype -- per-sample domain (region) ids; cast to
                ``long`` before the embedding lookup.

        Returns:
            torch.Tensor: Shape ``(batch_size, embed_dim)``, float32.
        """
        return self.embedding(domain_ids.long())


class DomainDualBranch(nn.Module):
    """Pairs an HR image encoder with a :class:`DomainEmbeddingBranch`.

    Mirrors :class:`CoordDualBranch` but conditions on the discrete domain label
    (``x["domain"]``) instead of geographic coordinates (``x["coords"]``).
    """

    def __init__(self, hr_encoder: Branch, lr_encoder: DomainEmbeddingBranch):
        """Store the HR image encoder and the domain-embedding encoder.

        Args:
            hr_encoder (Branch): Encoder applied to ``x["rgb"]``.
            lr_encoder (DomainEmbeddingBranch): Encoder applied to ``x["domain"]``.
        """
        super().__init__()
        self.hr_encoder = hr_encoder
        self.lr_encoder = lr_encoder

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode the HR image and the domain label independently.

        Args:
            x (Dict[str, torch.Tensor]): Batch dict with ``"rgb"``
                (``(batch_size, 3, H, W)``, float32) and ``"domain"``
                (``(batch_size,)``, integer -- domain/region id).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: ``(hr_features, lr_features)``,
            shapes ``(batch_size, hr_encoder.out_dim)`` and
            ``(batch_size, lr_encoder.out_dim)`` respectively, both float32.
            (Annotated as a single ``torch.Tensor`` in the signature, but a
            2-tuple is returned.)
        """
        hr_features = self.hr_encoder(x["rgb"])
        lr_features = self.lr_encoder(x["domain"])
        return hr_features, lr_features


if __name__ == "__main__":
    # Example usage
    hr_branch = DeitBranch(in_channels=3, pretrained=True)
    lr_branch = DeitBranch(in_channels=6, pretrained=True)
    model = DualBranch(hr_encoder=hr_branch, lr_encoder=lr_branch, landsat_channels=6)

    # Dummy input
    x = {
        "rgb": torch.randn(1, 3, 224, 224),
        "landsat": torch.randn(1, 6, 224, 224),
    }

    hr_features, lr_features = model(x)
    print("HR features shape:", hr_features.shape)
    print("LR features shape:", lr_features.shape)