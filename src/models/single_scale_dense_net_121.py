import torch.nn as nn
from torchvision.models import densenet121

class SingleScaleDenseNet121(nn.Module):
    def __init__(self, num_labels=62, image_net=True):
        super().__init__()

        weights = 'IMAGENET1K_V1' if image_net else None
        self.model = densenet121(weights=weights)
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, num_labels)
    
    def forward(self, x):
        rgb_processed = x['rgb']
        return self.model(rgb_processed)
