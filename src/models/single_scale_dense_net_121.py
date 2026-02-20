import torch.nn as nn
from torchvision.models import densenet121

class SingleScaleDenseNet121(nn.Module):
    def __init__(self, num_labels=62):
        super().__init__()

        self.model = densenet121(weights='IMAGENET1K_V1')
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, num_labels)
    
    def forward(self, x):
        rgb_processed = x['rgb']
        return self.model(rgb_processed)
