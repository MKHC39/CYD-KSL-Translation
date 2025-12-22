import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class FrozenResNetMeanClassifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = resnet18(weights=weights)

        # backbone up to avgpool
        self.backbone = nn.Sequential(*list(m.children())[:-1])  # -> (N,512,1,1)
        feat_dim = m.fc.in_features  # typically 512

        # freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, clips: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        clips:   (B,T,3,224,224) float32, ImageNet-normalised
        lengths: (B,) int64, number of valid frames per clip (<=T)
        returns: logits (B,num_classes)
        """
        B, T, C, H, W = clips.shape
        x = clips.view(B * T, C, H, W)

        f = self.backbone(x).flatten(1)          # (B*T,512)
        f = f.view(B, T, -1)                     # (B,T,512)

        # masked mean over time
        device = f.device
        t_idx = torch.arange(T, device=device).view(1, T)
        mask = (t_idx < lengths.view(B, 1)).float().unsqueeze(-1)  # (B,T,1)

        f_sum = (f * mask).sum(dim=1)            # (B,512)
        denom = mask.sum(dim=1).clamp(min=1.0)   # (B,1)
        f_mean = f_sum / denom                   # (B,512)

        return self.classifier(f_mean)
