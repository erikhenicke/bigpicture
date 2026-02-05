import torch
import torch.nn as nn
from transformers import ViTForImageClassification, AutoFeatureExtractor


class MultiResolutionDeiT(nn.Module):
    def __init__(self, num_labels=62):
        super().__init__()

        # Load feature extractor
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            'facebook/deit-tiny-patch16-224'
        )

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

        # Freeze feature extractor parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # Freeze encoder parameters
        for param in self.encoder_hr.parameters():
            param.requires_grad = False
        for param in self.encoder_lr.parameters():
            param.requires_grad = False
        
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
        
        # For Landsat 6 bands, we need to select 3 for RGB-like input, for pretrained usage.
        landsat = x['landsat'][:3, :, :]  # Take first 3 bands for now

        # Preprocess with feature extractor
        rgb_processed = self.feature_extractor(rgb, return_tensors='pt')['pixel_values']
        landsat_processed = self.feature_extractor(landsat, return_tensors='pt')['pixel_values']
 
        
        # Extract [CLS] features
        outputs_hr = self.encoder_hr.vit(rgb_processed)
        outputs_lr = self.encoder_lr.vit(landsat_processed)
        
        feat_hr = outputs_hr.last_hidden_state[:, 0, :]
        feat_lr = outputs_lr.last_hidden_state[:, 0, :]
        
        # Fuse and classify
        fused = self.fusion(torch.cat([feat_hr, feat_lr], dim=1))
        return self.classifier(fused)