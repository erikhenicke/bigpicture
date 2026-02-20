import torch
import torch.nn as nn
from torchvision.models import densenet121


class MultiScaleDenseNet121(nn.Module):
    def __init__(self, num_labels=62, in_channels=6):
        super().__init__()

        self.encoder_hr = densenet121(weights="IMAGENET1K_V1")
        self.encoder_lr = densenet121(weights="IMAGENET1K_V1")

        self.in_channels = in_channels
        if 4 <= in_channels <= 6:
            self._adapt_landsat_encoder_input_channels(in_channels=in_channels)
        elif in_channels == 3:
            pass
        else:
            raise ValueError(
                f"Unsupported number of input channels: {in_channels}. Supported values are 3, 4, 5, or 6."
            )

        num_features = self.encoder_hr.classifier.in_features

        self.encoder_hr.classifier = nn.Identity()
        self.encoder_lr.classifier = nn.Identity()

        self.fusion = nn.Sequential(
            nn.Linear(num_features * 2, num_features),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(num_features, num_features // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Linear(num_features // 2, num_labels)

    def forward(self, x, **kwargs):
        rgb = x["rgb"]
        landsat = x["landsat"][:, : self.in_channels, :, :]

        feat_hr = self._extract_features(self.encoder_hr, rgb)
        feat_lr = self._extract_features(self.encoder_lr, landsat)

        fused = self.fusion(torch.cat([feat_hr, feat_lr], dim=1))
        return self.classifier(fused)

    @staticmethod
    def _extract_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        features = model.features(x)
        out = nn.functional.relu(features, inplace=False)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        return torch.flatten(out, 1)

    def _adapt_landsat_encoder_input_channels(self, in_channels: int) -> None:
        old_conv = self.encoder_lr.features.conv0

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

        self.encoder_lr.features.conv0 = new_conv