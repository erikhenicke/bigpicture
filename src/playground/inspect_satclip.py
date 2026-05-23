from huggingface_hub import hf_hub_download
import os
import sys
sys.path.append(os.path.join(os.getcwd(), "lib/satclip/satclip")) 
from load import get_satclip
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

c = torch.randn(32, 2)  # Represents a batch of 32 locations (lon/lat)

model = get_satclip(
    hf_hub_download("microsoft/SatCLIP-ViT16-L10", "satclip-vit16-l10.ckpt"),
    hf_hub_download("microsoft/SatCLIP-ResNet18-L10", "satclip-resnet18-l10.ckpt"),
    hf_hub_download("microsoft/SatCLIP-ResNet50-L10", "satclip-resnet50-l10.ckpt"),
    device=device,
)  # Only loads location encoder by default
model.eval()
with torch.no_grad():
    emb = model(c.double().to(device)).detach().cpu()
    print(emb.shape)