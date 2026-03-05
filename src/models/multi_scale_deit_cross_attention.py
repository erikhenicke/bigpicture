import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig


class CrossAttentionBlock(nn.Module):
    def __init__(
        self, dim=192, num_heads=3, attn_drop=0.1, proj_drop=0.1 
    ):
        super().__init__()

        self.norm_q_hr = nn.LayerNorm(dim)
        self.norm_q_lr = nn.LayerNorm(dim)
        self.norm_kv_hr = nn.LayerNorm(dim)
        self.norm_kv_lr = nn.LayerNorm(dim)

        self.cross_attn_hr = nn.MultiheadAttention(
            dim, num_heads, dropout=attn_drop, batch_first=True
        )
        self.cross_attn_lr = nn.MultiheadAttention(
            dim, num_heads, dropout=attn_drop, batch_first=True
        )


    def forward(self, tokens_hr, tokens_lr):
        cls_hr = tokens_hr[:, 0, :]
        cls_lr = tokens_lr[:, 0, :]
        patch_hr = tokens_hr[:, 0:, :]
        patch_lr = tokens_lr[:, 0:, :]

        delta_hr, _ = self.cross_attn_hr(
            query=self.norm_q_hr(cls_hr),
            key=self.norm_kv_lr(tokens_lr),
            value=self.norm_kv_lr(tokens_lr),
            need_weights=False,
        )
        cls_hr_out = cls_hr + delta_hr

        delta_lr, _ = self.cross_attn_lr(
            query=self.norm_q_lr(cls_lr),
            key=self.norm_kv_hr(tokens_hr),
            value=self.norm_kv_hr(tokens_hr),
            need_weights=False,
        )
        cls_lr_out = cls_lr + delta_lr

        return torch.cat([cls_hr_out, patch_hr], dim=1), torch.cat(
            [cls_lr_out, patch_lr], dim=1
        )


class DualEncoderWithCrossAttention(nn.Module):
    def __init__(
        self,
        encoder_hr,
        encoder_lr,
        cross_attn_depths=None,
        cross_attn_dim=192,
        cross_attn_heads=3,
        cross_attn_drop=0.0,
    ):
        super().__init__()

        self.encoder_hr = encoder_hr
        self.encoder_lr = encoder_lr
        self.cross_attn_depths = cross_attn_depths or [11]

        self.cross_attn_blocks = nn.ModuleDict()
        for depth in self.cross_attn_depths:
            self.cross_attn_blocks[f"depth_{depth}"] = CrossAttentionBlock(
                dim=cross_attn_dim,
                num_heads=cross_attn_heads,
                attn_drop=cross_attn_drop,
                proj_drop=cross_attn_drop,
            )

    def forward(self, rgb, landsat):
        vit_hr = self.encoder_hr.vit
        vit_lr = self.encoder_lr.vit

        x_hr = vit_hr.embeddings(rgb)
        x_lr = vit_lr.embeddings(landsat)

        for layer_idx, block_hr in enumerate(vit_hr.encoder.layer):
            x_hr = block_hr(x_hr)[0]
            x_lr = vit_lr.encoder.layer[layer_idx](x_lr)[0]

            if layer_idx in self.cross_attn_depths:
                x_hr, x_lr = self.cross_attn_blocks[f"depth_{layer_idx}"](x_hr, x_lr)

        return vit_hr.layernorm(x_hr), vit_lr.layernorm(x_lr)


class MultiScaleDeiTCrossFusion(nn.Module):
    def __init__(
        self,
        num_labels=62,
        in_channels=6,
        pos_enc="learnable",
        image_net="both",
        cross_attn_depths=[11],
        cross_attn_enabled=True,
        cross_fusion_dim=192,
        cross_fusion_heads=3,
        cross_fusion_drop=0.0,
    ):
        super().__init__()

        if not (3 <= in_channels <= 6):
            raise ValueError(f"Unsupported number of input channels: {in_channels}.")

        self.in_channels = in_channels
        self.cross_attn_enabled = cross_attn_enabled

        if image_net == "both":
            self.encoder_hr = self._get_deit(num_labels=num_labels, pretrained=True)
            self.encoder_lr = self._get_deit(num_labels=num_labels, pretrained=True)
        elif image_net == "hr":
            self.encoder_hr = self._get_deit(num_labels=num_labels, pretrained=True)
            self.encoder_lr = self._get_deit(num_labels=num_labels, pretrained=False)
        else:  # 'image_net' == 'none'
            self.encoder_hr = self._get_deit(num_labels=num_labels, pretrained=False)
            self.encoder_lr = self._get_deit(num_labels=num_labels, pretrained=False)

        if 4 <= in_channels <= 6:
            self._adapt_landsat_encoder_input_channels(in_channels)

        self.encoder_hr.classifier = nn.Identity()
        self.encoder_lr.classifier = nn.Identity()

        if pos_enc == "sinusoidal":
            self._set_sinusoidal_positional_encoding(self.encoder_hr.vit)
            self._set_sinusoidal_positional_encoding(self.encoder_lr.vit)

        if self.cross_attn_enabled:
            self.dual_encoder = DualEncoderWithCrossAttention(
                self.encoder_hr,
                self.encoder_lr,
                cross_attn_depths=cross_attn_depths,
                cross_attn_dim=cross_fusion_dim,
                cross_attn_heads=cross_fusion_heads,
                cross_attn_drop=cross_fusion_drop,
            )

        self.fusion = nn.Sequential(
            nn.Linear(cross_fusion_dim * 2, cross_fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cross_fusion_dim * 2, cross_fusion_dim),
        )
        self.classifier = nn.Linear(cross_fusion_dim, num_labels)

    def forward(self, x, **kwargs):
        rgb = x["rgb"]
        landsat = x["landsat"][:, : self.in_channels, :, :]

        if self.cross_attn_enabled:
            tokens_hr, tokens_lr = self.dual_encoder(rgb, landsat)
        else:
            tokens_hr = self.encoder_hr.vit(rgb).last_hidden_state
            tokens_lr = self.encoder_lr.vit(landsat).last_hidden_state

        feat_hr = tokens_hr[:, 0, :]
        feat_lr = tokens_lr[:, 0, :]

        fused = self.fusion(torch.cat([feat_hr, feat_lr], dim=1))
        return self.classifier(fused)

    def _adapt_landsat_encoder_input_channels(self, in_channels: int) -> None:
        patch_embeddings = self.encoder_lr.vit.embeddings.patch_embeddings
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
            new_projection.weight[:, : old_projection.in_channels, :, :] = (
                old_projection.weight
            )
            if in_channels > old_projection.in_channels:
                extra_channels = in_channels - old_projection.in_channels
                mean_weight = old_projection.weight.mean(dim=1, keepdim=True)
                new_projection.weight[:, old_projection.in_channels :, :, :] = (
                    mean_weight.repeat(1, extra_channels, 1, 1)
                )
            if old_projection.bias is not None:
                new_projection.bias.copy_(old_projection.bias)

        patch_embeddings.projection = new_projection
        patch_embeddings.num_channels = in_channels
        self.encoder_lr.config.num_channels = in_channels
        self.encoder_lr.vit.config.num_channels = in_channels

    def _set_sinusoidal_positional_encoding(self, vit_module) -> None:
        embeddings = vit_module.embeddings
        position_embeddings = embeddings.position_embeddings
        seq_len = position_embeddings.shape[1]
        hidden_size = position_embeddings.shape[2]

        sinusoidal = self._build_sinusoidal_encoding(
            seq_len, hidden_size, position_embeddings.device
        )
        embeddings.position_embeddings = nn.Parameter(sinusoidal, requires_grad=False)

    @staticmethod
    def _build_sinusoidal_encoding(
        seq_len: int, hidden_size: int, device: torch.device
    ) -> torch.Tensor:
        position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(
            1
        )
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2, dtype=torch.float32, device=device)
            * (-torch.log(torch.tensor(10000.0, device=device)) / hidden_size)
        )

        encoding = torch.zeros(seq_len, hidden_size, dtype=torch.float32, device=device)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])

        return encoding.unsqueeze(0)

    @staticmethod
    def _get_deit(
        num_labels: int, pretrained: bool = True
    ) -> ViTForImageClassification:
        if pretrained:
            model = ViTForImageClassification.from_pretrained(
                "facebook/deit-tiny-patch16-224",
                output_hidden_states=True,
                num_labels=num_labels,
                ignore_mismatched_sizes=True,
                use_safetensors=True,
            )
        else:
            config = ViTConfig.from_pretrained("facebook/deit-tiny-patch16-224")
            config.output_hidden_states = True
            config.num_labels = num_labels
            model = ViTForImageClassification(config)

        return model
