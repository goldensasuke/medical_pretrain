import torch
import torch.nn as nn
from torchvision import models

def get_model(model_name='efficientnet_b0', pretrained=True):
    if hasattr(models, model_name):
        model = getattr(models, model_name)(weights='IMAGENET1K_V1' if pretrained else None)
    else:
        raise ValueError(f"Model {model_name} not found in torchvision.models")
    return model

class MaskedAutoencoder(nn.Module):
    def __init__(self, encoder, img_size=224, patch_size=16, mask_ratio=0.25):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        self.img_size = img_size
        self.patch_size = patch_size
        self.decoder = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, img_size*img_size*3)
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        mask = torch.rand_like(x) < self.mask_ratio
        x_masked = x.clone()
        x_masked[mask] = 0
        encoded = self.encoder(x_masked)
        decoded = self.decoder(encoded)
        decoded = decoded.view(B, C, H, W)
        return decoded
