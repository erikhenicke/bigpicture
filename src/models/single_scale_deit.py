import torch.nn as nn
from transformers import ViTForImageClassification, AutoFeatureExtractor

class SingleScaleDeiT(nn.Module):
    def __init__(self, num_labels=62):
        super().__init__()

        # Load feature extractor
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            'facebook/deit-tiny-patch16-224'
        )

        # Load DeiT model
        self.model = ViTForImageClassification.from_pretrained(
            'facebook/deit-tiny-patch16-224',
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )

        # Freeze feature extractor parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # Freeze encoder parameters
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, x, **kwargs):
        # Only use RGB for baseline
        rgb_processed = self.feature_extractor(x['rgb'], return_tensors='pt')['pixel_values']
        return self.model(rgb_processed).logits