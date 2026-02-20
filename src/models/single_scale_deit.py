import torch.nn as nn
from transformers import ViTForImageClassification 

class SingleScaleDeiT(nn.Module):
    def __init__(self, num_labels=62):
        super().__init__()

        # Load DeiT model
        self.model = ViTForImageClassification.from_pretrained(
            'facebook/deit-tiny-patch16-224',
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )

    def forward(self, x, **kwargs):
        # Only use RGB for baseline
        rgb_processed = x['rgb']
        return self.model(rgb_processed).logits