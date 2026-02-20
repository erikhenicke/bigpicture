import torch
import torch.nn as nn
from transformers import ViTForImageClassification 


class MultiScaleDeiT(nn.Module):
    def __init__(self, num_labels=62, in_channels=6, pos_enc='learnable'):
        super().__init__()

        # Independent encoders
        self.encoder_hr = ViTForImageClassification.from_pretrained(
            'facebook/deit-tiny-patch16-224',
            output_hidden_states=True,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        self.encoder_lr = ViTForImageClassification.from_pretrained(
            'facebook/deit-tiny-patch16-224',
            output_hidden_states=True,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )

        if pos_enc not in ('learnable', 'sinusoidal'):
            raise ValueError(
                f"Unsupported positional encoding: {pos_enc}. Supported values are 'learnable' or 'sinusoidal'."
            )
        self.pos_enc = pos_enc
        if self.pos_enc == 'sinusoidal':
            self._set_sinusoidal_positional_encoding(self.encoder_hr.vit)
            self._set_sinusoidal_positional_encoding(self.encoder_lr.vit)

        self.in_channels = in_channels
        if 4 <= in_channels <= 6: 
            self._adapt_landsat_encoder_input_channels(in_channels=in_channels)
        elif in_channels == 3:
            pass
        else:
            raise ValueError(f"Unsupported number of input channels: {in_channels}. Supported values are 3, 4, 5, or 6.")
        
        # Remove classifiers
        self.encoder_hr.classifier = nn.Identity()
        self.encoder_lr.classifier = nn.Identity()
        
        # Fusion (DeiT-tiny hidden_size=192)
        self.fusion = nn.Sequential(
            nn.Linear(192 * 2, 384),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(384, 192)
        )
        self.classifier = nn.Linear(192, num_labels)
    
    def forward(self, x, **kwargs):
        # x is dict with 'rgb' and 'landsat'
        rgb = x['rgb']
        landsat = x['landsat'][:, :self.in_channels, :, :]  # Take only the specified number of channels (first dim is batch size)

        # Extract [CLS] features
        outputs_hr = self.encoder_hr.vit(rgb)
        outputs_lr = self.encoder_lr.vit(landsat)
        
        # last_hidden_state shape: (batch_size, seq_len, hidden_size), we take the [CLS] token at index 0
        feat_hr = outputs_hr.last_hidden_state[:, 0, :]
        feat_lr = outputs_lr.last_hidden_state[:, 0, :]
        
        # Fuse and classify
        fused = self.fusion(torch.cat([feat_hr, feat_lr], dim=1))
        return self.classifier(fused)

    def _adapt_landsat_encoder_input_channels(self, in_channels: int) -> None:
        patch_embeddings = self.encoder_lr.vit.embeddings.patch_embeddings
        old_projection = patch_embeddings.projection

        if old_projection.in_channels == in_channels:
            return

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

    def _set_sinusoidal_positional_encoding(self, vit_module) -> None:
        embeddings = vit_module.embeddings
        position_embeddings = embeddings.position_embeddings
        seq_len = position_embeddings.shape[1]
        hidden_size = position_embeddings.shape[2]

        sinusoidal = self._build_sinusoidal_encoding(seq_len, hidden_size, position_embeddings.device)
        embeddings.position_embeddings = nn.Parameter(sinusoidal, requires_grad=False)

    @staticmethod
    def _build_sinusoidal_encoding(seq_len: int, hidden_size: int, device: torch.device) -> torch.Tensor:
        """Generates a sinusoidal positional encoding matrix of shape (1, seq_len, hidden_size).
        
        Reference: "Attention Is All You Need" (Vaswani et al., 2017) - Section 3.5
            and for explanation: https://iclr-blogposts.github.io/2025/blog/positional-embedding/
        """
        position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2, dtype=torch.float32, device=device)
            * (-torch.log(torch.tensor(10000.0, device=device)) / hidden_size)
        )

        encoding = torch.zeros(seq_len, hidden_size, dtype=torch.float32, device=device)
        # Even indices: sin, Odd indices: cos
        encoding[:, 0::2] = torch.sin(position * div_term)
        # If hidden_size is odd, the last dimension will be even, so we need to ensure we don't go out of bounds.
        encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])

        return encoding.unsqueeze(0)