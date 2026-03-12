from train.run_experiment import get_data_loader

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import ViTForImageClassification

from load import get_satclip


class MultiScaleDeiTSatCLIP(nn.Module):
    def __init__(
        self,
        num_labels: int = 62,
        in_channels: int = 6,
        satclip_repo_id: str = "microsoft/SatCLIP-ViT16-L10",
        satclip_ckpt_name: str = "satclip-vit16-l10.ckpt",
    ) -> None:
        super().__init__()

        if not (in_channels == 3 or in_channels == 6):
            raise ValueError(
                f"Unsupported number of input channels: {in_channels}. Supported values are 3 or 6."
            )
        self.in_channels = in_channels

        # FMoW (high-resolution) branch - trainable
        self.encoder_hr = ViTForImageClassification.from_pretrained(
            "facebook/deit-tiny-patch16-224",
            output_hidden_states=True,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            use_safetensors=True,
        )
        self.encoder_hr.classifier = nn.Identity()

        # SatCLIP (low-resolution) branch - frozen
        satclip_root = self._load_satclip_root_model(satclip_repo_id, satclip_ckpt_name)
        self.encoder_lr = self._extract_visual_encoder(satclip_root)

        self._adapt_satclip_input_channels(self.in_channels)

        self._freeze_satclip_branch()

        self.satclip_projection = nn.Linear(256, 192)

        # Dynamic fusion size because SatCLIP feature dim may differ from DeiT
        self.fusion = nn.Sequential(
            nn.Linear(384, 384),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(384, 192),
        )
        self.classifier = nn.Linear(192, num_labels)

    def forward(self, x: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        rgb = x["rgb"]
        landsat = x["landsat"][:, : self.in_channels, :, :]

        outputs_hr = self.encoder_hr.vit(rgb)
        feat_hr = outputs_hr.last_hidden_state[:, 0, :]

        outputs_lr = self.encoder_lr(landsat)
        feat_lr = self.satclip_projection(outputs_lr).to(dtype=feat_hr.dtype)

        fused = self.fusion(torch.cat([feat_hr, feat_lr], dim=1))
        return self.classifier(fused)

    def _freeze_satclip_branch(self) -> None:
        for name, p in self.encoder_lr.named_parameters():
            if "patch_embed.proj" not in name:
                p.requires_grad = False

    def _adapt_satclip_input_channels(self, in_channels: int) -> None:
        proj_path, old_projection = self._find_patch_projection(self.encoder_lr)

        new_projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_projection.out_channels,
            kernel_size=old_projection.kernel_size,
            stride=old_projection.stride,
            padding=old_projection.padding,
            dilation=old_projection.dilation,
            groups=old_projection.groups,
            bias=old_projection.bias is not None,
        )

        with torch.no_grad():
            # LandSat has bands in the order: [B, G, R, NIR, SWIR1, SWIR2] and SatCLIP was trained on Sentinel-2 with bands B1-B12 with B2, B3, B4, B7, B11, B12 corresponding to the LandSat bands.
            if in_channels == 3:
                new_projection.weight[:, :3, :, :] = old_projection.weight[:, 1:4, :, :]
            else: # in_channels == 6
                new_projection.weight[:, :, :, :] = old_projection.weight[:, [1, 2, 3, 7, 10, 11], :, :]

            if old_projection.bias is not None:
                new_projection.bias.copy_(old_projection.bias)

        self._set_attr_by_path(self.encoder_lr, proj_path, new_projection)
        self.encoder_lr.in_chans = in_channels

    @classmethod
    def _load_satclip_root_model(cls, repo_id: str, ckpt_name: str) -> Any:
        ckpt_path = hf_hub_download(repo_id, ckpt_name)
        return get_satclip(ckpt_path, device="cpu", return_all=True)

    @classmethod
    def _extract_visual_encoder(cls, model: Any) -> Optional[nn.Module]:
        return cls._get_attr_by_path(model, "visual")

    @classmethod
    def _find_patch_projection(cls, root: nn.Module) -> Tuple[str, nn.Conv2d]:
            path = "patch_embed.proj"  
            mod = cls._get_attr_by_path(root, path) 
            if isinstance(mod, nn.Conv2d):
                return path, mod

    @staticmethod
    def _get_attr_by_path(root: Any, path: str) -> Any:
        obj = root
        for part in path.split("."):
            if not hasattr(obj, part):
                return None
            obj = getattr(obj, part)
        return obj

    @staticmethod
    def _set_attr_by_path(root: Any, path: str, value: Any) -> None:
        parts = path.split(".")
        parent = root
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], value)

def sanity_check(model):
    model.zero_grad(set_to_none=True)
    model.train()

    feat = model({
        "rgb": torch.randn(2, 3, 224, 224),
        "landsat": torch.randn(2, 6, 224, 224),
    })
    loss = feat.sum()
    loss.backward()

    for n, p in model.encoder_lr.named_parameters():
        if "patch_embed.proj" in n:
            assert p.grad is not None, f"Missing grad: {n}"
        else:
            assert p.grad is None, f"Unexpected grad: {n}"

if __name__ == "__main__":
    model = MultiScaleDeiTSatCLIP(in_channels=6)
    sanity_check(model)