import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig


class SingleScaleDeiTLandsat(nn.Module):
    def __init__(self, num_labels=62, in_channels=6, image_net=True):
        super().__init__()

        if not (3 <= in_channels <= 6):
            raise ValueError(
                f"Unsupported number of input channels: {in_channels}. Supported values are 3, 4, 5, or 6."
            )

        self.in_channels = in_channels

        # Load DeiT model
        if image_net:
            self.model = ViTForImageClassification.from_pretrained(
                'facebook/deit-tiny-patch16-224',
                num_labels=num_labels,
                ignore_mismatched_sizes=True,
                use_safetensors=True,
            )

            if in_channels != 3:
                self._adapt_input_channels(in_channels)
        else:
            self.model = ViTForImageClassification(ViTConfig.from_pretrained(
                'facebook/deit-tiny-patch16-224',
                num_labels=num_labels,
                num_channels=in_channels,
            ))

    def forward(self, x, **kwargs):
        # Only use Landsat for baseline
        landsat = x['landsat']
        if landsat.shape[1] < self.in_channels:
            raise ValueError(
                f"Expected at least {self.in_channels} channels in 'landsat', got {landsat.shape[1]}."
            )

        landsat_processed = landsat[:, :self.in_channels, :, :]
        return self.model(landsat_processed).logits

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
