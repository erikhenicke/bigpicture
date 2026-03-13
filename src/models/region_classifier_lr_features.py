"""
Region Classifier using LR (Landsat) Encoder Features

This model loads a pretrained MultiScaleDeiTCrossFusion model and extracts
the encoder_lr (low-resolution/Landsat) features to predict continental regions.
The encoder_lr features (CLS token) are passed through a classification head
to predict one of 6 regions: Europe, Americas, Asia, Africa, Oceania, Unknown.
"""

import torch
import torch.nn as nn


class RegionClassifierLRFeatures(nn.Module):
    """
    Standalone region classifier that extracts features from the encoder_lr
    (Landsat encoder) of a pretrained MultiScaleDeiTCrossFusion model.
    
    The model loads a checkpoint containing the full multi-scale model,
    freezes the encoder_lr, and adds a region classification head.
    """
    
    # Region names (6 regions: 5 continents + other)
    REGION_NAMES = ['Europe', 'Americas', 'Asia', 'Africa', 'Oceania', 'Other']
    NUM_REGIONS = 6
    ENCODER_DIM = 192  # DeiT-tiny hidden dimension
    
    def __init__(
        self,
        num_regions: int = 6,
        encoder_dim: int = 192,
        hidden_dim: int = 256,
        dropout_rate: float = 0.2,
    ):
        """
        Args:
            num_regions: Number of regions to classify (default: 6)
            encoder_dim: Dimension of encoder output features (default: 192 for DeiT-tiny)
            hidden_dim: Hidden dimension for the classification head
            dropout_rate: Dropout rate in classification head
        """
        super().__init__()
        
        self.num_regions = num_regions
        self.encoder_dim = encoder_dim
        
        # Classification head: takes encoder_dim features and predicts num_regions classes
        self.region_classifier = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_regions),
        )
    
    def forward(self, feat_lr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_lr: LR encoder features of shape (batch_size, encoder_dim)
                     Typically the CLS token from encoder_lr
        
        Returns:
            Region logits of shape (batch_size, num_regions)
        """
        return self.region_classifier(feat_lr)
    
    @staticmethod
    def extract_encoder_lr_features(
        model: nn.Module,
        x: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Extract encoder_lr (Landsat) features from a multi-scale model.
        
        Supports:
        - MultiScaleDeiTCrossFusion: Uses cross-attention aware forward pass
        - MultiScaleDeiT: Uses standard ViT forward pass
        - MultiScaleDenseNet121: Uses encoder_lr feature extraction
        
        Args:
            model: Multi-scale model with encoder_lr
            x: Input dict with 'rgb' and 'landsat' tensors
        
        Returns:
            CLS token features from encoder_lr of shape (batch_size, feature_dim)
        """

        rgb = x['rgb']
        landsat = x['landsat'][:, :model.in_channels, :, :]
        
        model_type = type(model).__name__
        
        if model_type == 'MultiScaleDeiTCrossFusion':

            if model.cross_attn_enabled:
                _, tokens_lr = model.dual_encoder(rgb, landsat)
            else:
                tokens_lr = model.encoder_lr.vit(landsat).last_hidden_state
            
            feat_lr = tokens_lr[:, 0, :]
        
        elif model_type == 'MultiScaleDeiT':

            output_lr = model.encoder_lr.vit(landsat)
            feat_lr = output_lr.last_hidden_state[:, 0, :]
        
        elif model_type == 'MultiScaleDeiTSatCLIP':

            output_lr = model.encoder_lr(landsat) 
            feat_lr = model.satclip_projection(output_lr)
        
        else:
            raise NotImplementedError(
                f"Feature extraction not implemented for {model_type}. "
                f"Supported models: MultiScaleDeiTCrossFusion, MultiScaleDeiT, MultiScaleDeiTSatCLIP"
            )
        
        return feat_lr
    
    @classmethod
    def from_pretrained_multiscale_model(
        cls,
        model: nn.Module,
        **kwargs,
    ) -> 'RegionClassifierLRFeatures':
        """
        Create a region classifier from a pretrained multi-scale model.
        
        Args:
            model: Pretrained multi-scale model (e.g., MultiScaleDeiTCrossFusion)
            freeze_encoder: Whether to freeze the encoder_lr during training
            **kwargs: Additional arguments for RegionClassifierLRFeatures.__init__
        
        Returns:
            RegionClassifierLRFeatures instance
        """

        if hasattr(model.encoder_lr, 'config'):
            encoder_dim = model.encoder_lr.config.hidden_size
        else:
            encoder_dim = kwargs.pop('encoder_dim', cls.ENCODER_DIM)
        
        region_classifier = cls(
            encoder_dim=encoder_dim,
            **kwargs
        )
        
        region_classifier.encoder_lr = model.encoder_lr
        
        for param in model.encoder_lr.parameters():
            param.requires_grad = False
        
        return region_classifier