# TODO: Implement SatClip and CrossAttention

import torch
import torch.nn as nn
from abc import abstractmethod
from typing import Dict


class Branch(nn.Module):    
    def __init__(self, in_channels=3, **kwargs):
        super().__init__()

        if not (3 <= in_channels <= 6):
            raise ValueError(f"Unsupported number of input channels: {in_channels}. Supported values are 3, 4, 5, or 6.")

        self.in_channels = in_channels
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


class DenseNetBranch(Branch):
    def __init__(self, in_channels=3, image_net=True):
        super().__init__(in_channels=in_channels, image_net=image_net)
    
    def forward(self, x):
        features = self.model.features(x)
        out = nn.functional.relu(features, inplace=False)
        # Adaptive average pooling sets kernel size and stride automatically, with (1, 1) global average pooling is applied to each feature map.
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        return torch.flatten(out, 1)

    @property
    def out_dim(self) -> int:
        return self.model.classifier.in_features

    def _get_model(self, image_net=True):
        from torchvision.models import densenet121
        weights = 'IMAGENET1K_V1' if image_net else None
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
            new_conv.weight[:, : old_conv.in_channels, :, :] = old_conv.weight
            if in_channels > old_conv.in_channels:
                extra_channels = in_channels - old_conv.in_channels
                mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
                new_conv.weight[:, old_conv.in_channels :, :, :] = mean_weight.repeat(
                    1, extra_channels, 1, 1
                )
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        self.model.features.conv0 = new_conv 

class DeitBranch(Branch):
    def __init__(self, in_channels=3, image_net=True):
        super().__init__(in_channels=in_channels, image_net=image_net)

    def forward(self, x):
        tokens = self.model.vit(x).last_hidden_state
        return tokens[:, 0, :]

    @property
    def out_dim(self) -> int:
        return self.model.config.hidden_size

    def _get_model(self, image_net=True):
        from transformers import ViTForImageClassification, ViTConfig

        if image_net:
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
            new_projection.weight[:, :old_projection.in_channels, :, :] = old_projection.weight
            if in_channels > old_projection.in_channels:
                extra_channels = in_channels - old_projection.in_channels
                mean_weight = old_projection.weight.mean(dim=1, keepdim=True)
                new_projection.weight[:, old_projection.in_channels:, :, :] = mean_weight.repeat(
                    1, extra_channels, 1, 1
                )
            if old_projection.bias is not None:
                new_projection.bias.copy_(old_projection.bias)

        patch_embeddings.projection = new_projection
        patch_embeddings.num_channels = in_channels
        self.model.config.num_channels = in_channels
        self.model.vit.config.num_channels = in_channels


class DualBranch(nn.Module):
    def __init__(self, hr_encoder: Branch, lr_encoder: Branch, landsat_channels: int = 6):
        super().__init__()

        if lr_encoder.in_channels != landsat_channels:
            raise ValueError(
                f"LR encoder input channels ({lr_encoder.in_channels}) do not match specified landsat_channels ({landsat_channels})."
            )

        self.hr_encoder = hr_encoder
        self.lr_encoder = lr_encoder
        self.landsat_channels = landsat_channels

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        hr_image = x["rgb"]
        lr_image = x["landsat"][:, : self.landsat_channels, :, :]
        hr_features = self.hr_encoder(hr_image)
        lr_features = self.lr_encoder(lr_image)
        return hr_features, lr_features 

if __name__ == "__main__":
    # Example usage
    hr_branch = DeitBranch(in_channels=3, image_net=True)
    lr_branch = DeitBranch(in_channels=6, image_net=True)
    model = DualBranch(hr_encoder=hr_branch, lr_encoder=lr_branch, landsat_channels=6)

    # Dummy input
    x = {
        "rgb": torch.randn(1, 3, 224, 224),
        "landsat": torch.randn(1, 6, 224, 224),
    }

    hr_features, lr_features = model(x)
    print("HR features shape:", hr_features.shape)
    print("LR features shape:", lr_features.shape)