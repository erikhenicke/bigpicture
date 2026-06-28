import os
import sys

import torch
import torch.nn as nn
from abc import abstractmethod
from typing import Dict, Optional

from models.components.spatial_encoding import SpatialEncoding


VALID_EXTRA_CH_INITS = ("average", "zero", "he")


class Branch(nn.Module):
    def __init__(self, in_channels: int = 3, landsat_channel_init: str = "zero", stacked: bool = False, bgr_input: bool = False, **kwargs):
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
        pass

    @abstractmethod
    def _adapt_input_channels(self, in_channels: int) -> None:
        pass

    @property
    @abstractmethod
    def out_dim(self) -> int:
        pass

    def _reorder_pretrained_weights(self, weight: torch.Tensor) -> torch.Tensor:
        if self.bgr_input:
            return weight[:, [2, 1, 0], :, :]
        return weight

    def _init_extra_weights(self, weight_slice: torch.Tensor, old_weight: torch.Tensor) -> None:
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
    def __init__(self, in_channels: int = 3, pretrained: bool = True, landsat_channel_init: str = "zero", stacked: bool = False, bgr_input: bool = False):
        super().__init__(in_channels=in_channels, landsat_channel_init=landsat_channel_init, stacked=stacked, bgr_input=bgr_input, pretrained=pretrained)
    
    def forward(self, x):
        features = self.model.features(x)
        out = nn.functional.relu(features, inplace=False)
        # Adaptive average pooling sets kernel size and stride automatically, with (1, 1) global average pooling is applied to each feature map.
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        return torch.flatten(out, 1)

    @property
    def out_dim(self) -> int:
        return self.model.classifier.in_features

    def _get_model(self, pretrained=True):
        from torchvision.models import densenet121
        weights = 'IMAGENET1K_V1' if pretrained else None
        return densenet121(weights=weights)

    def _adapt_input_channels(self, in_channels):
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
    def __init__(self, in_channels: int = 3, pretrained: bool = True, landsat_channel_init: str = "zero", stacked: bool = False, bgr_input: bool = False):
        super().__init__(in_channels=in_channels, landsat_channel_init=landsat_channel_init, stacked=stacked, bgr_input=bgr_input, pretrained=pretrained)

    def forward(self, x):
        tokens = self.model.vit(x).last_hidden_state
        return tokens[:, 0, :]

    @property
    def out_dim(self) -> int:
        return self.model.config.hidden_size

    def _get_model(self, pretrained=True):
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
        super().__init__(in_channels=in_channels, landsat_channel_init=landsat_channel_init, stacked=stacked, bgr_input=bgr_input, model_name=model_name, pretrained=pretrained)

    def forward(self, x):
        return self.model(x)

    @property
    def out_dim(self) -> int:
        return self.model.num_features

    def _get_model(self, model_name: str, pretrained: bool = True) -> nn.Module:
        import timm

        return timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=self.in_channels,
            num_classes=0,
            global_pool="avg",
        )

    def _adapt_input_channels(self, in_channels: int) -> None:
        # timm has already built the stem with the requested in_chans and
        # copied pretrained weights (averaging / repeating as needed), so the
        # base-class hook has nothing left to do.
        return


class DualBranch(nn.Module):
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
        super().__init__()

        self.hr_encoder = hr_encoder
        self.lr_encoder = lr_encoder
        self.landsat_channels = landsat_channels
        self.coord_channels_hr = coord_channels_hr
        self.coord_channels_lr = coord_channels_lr
        self.hr_spatial_encoding = hr_spatial_encoding
        self.lr_spatial_encoding = lr_spatial_encoding

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
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
        super().__init__(in_channels=in_channels, landsat_channel_init=landsat_channel_init, stacked=stacked, bgr_input=bgr_input, pretrained=pretrained)
        if pretrained:
            if unfreeze_all:
                self.model.requires_grad_(True)
            else:
                self._freeze(unfreeze_first_block, unfreeze_head, num_unfreeze_last)

    def _freeze(self, unfreeze_first_block: bool, unfreeze_head: bool, num_unfreeze_last: int) -> None:
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
        return self.model(x)

    @property
    def out_dim(self) -> int:
        return self.model.head.out_features

    def _get_model(self, pretrained=True):
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
        pass


class DINOv3Branch(Branch):
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
        super().__init__(in_channels=in_channels, landsat_channel_init=landsat_channel_init, stacked=stacked, bgr_input=bgr_input, pretrained=pretrained, model_size=model_size)

        if freeze:
            self.model.requires_grad_(False)

    def forward(self, x):
        return self.model(x).pooler_output

    @property
    def out_dim(self) -> int:
        return self.model.config.hidden_size

    def _get_model(self, pretrained: bool = True, model_size: str = "base") -> nn.Module:
        from transformers import AutoModel, AutoConfig

        if pretrained:
            if model_size == "base":
                return AutoModel.from_pretrained(self.HF_MODEL_BASE)
            else:
                return AutoModel.from_pretrained(self.HF_MODEL_LARGE)

        config = AutoConfig.from_pretrained(self.HF_MODEL_NAME)
        return AutoModel.from_config(config)

    def _adapt_input_channels(self, in_channels: int) -> None:
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
    def __init__(
        self,
        embed_dim: int = 512,
        legendre_polys: int = 10,
        capacity: int = 512,
        num_hidden_layers: int = 2,
    ):
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
        return self._embed_dim

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        embeddings = self.location_encoder(coords.double())
        return embeddings.float()


class CoordDualBranch(nn.Module):
    def __init__(self, hr_encoder: Branch, lr_encoder: SatCLIPLocationBranch):
        super().__init__()
        self.hr_encoder = hr_encoder
        self.lr_encoder = lr_encoder

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
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
        super().__init__()
        self.embedding = nn.Embedding(num_domains, embed_dim)
        self._embed_dim = embed_dim

    @property
    def out_dim(self) -> int:
        return self._embed_dim

    def forward(self, domain_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(domain_ids.long())


class DomainDualBranch(nn.Module):
    """Pairs an HR image encoder with a :class:`DomainEmbeddingBranch`.

    Mirrors :class:`CoordDualBranch` but conditions on the discrete domain label
    (``x["domain"]``) instead of geographic coordinates (``x["coords"]``).
    """

    def __init__(self, hr_encoder: Branch, lr_encoder: DomainEmbeddingBranch):
        super().__init__()
        self.hr_encoder = hr_encoder
        self.lr_encoder = lr_encoder

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
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