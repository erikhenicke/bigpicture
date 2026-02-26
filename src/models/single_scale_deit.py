import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig

class SingleScaleDeiT(nn.Module):
    def __init__(self, num_labels=62, image_net=True):
        super().__init__()

        # Load DeiT model
        if image_net:
            self.model = ViTForImageClassification.from_pretrained(
                'facebook/deit-tiny-patch16-224',
                num_labels=num_labels,
                ignore_mismatched_sizes=True,
                use_safetensors=True,
            )
        else:
            self.model = ViTForImageClassification(ViTConfig.from_pretrained(
                'facebook/deit-tiny-patch16-224',
                num_labels=num_labels,
                ignore_mismatched_sizes=True,
                use_safetensors=True,
            ))

    def forward(self, x, **kwargs):
        # Only use RGB for baseline
        rgb_processed = x['rgb']
        return self.model(rgb_processed).logits
