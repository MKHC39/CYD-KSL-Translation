import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from preprocess.preprocess_clip import preprocess_stem

class ResNet18(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = resnet18(weights=weights)

        # Everything except the final fc layer
        self.features = nn.Sequential(*list(m.children())[:-1])  # ends at avgpool
        self.out_dim = 512  # ResNet18 embedding size (post-avgpool) (see torchvision resnet docs)
        # (The 512 detail is commonly known; if you want, we can assert it at runtime.)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,3,224,224)
        y = self.features(x)          # (N,512,1,1)
        y = y.flatten(1)              # (N,512)
        return y

def main ():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = ResNet18(pretrained=True).to(device)
    backbone.eval()  # inference mode for feature extraction

    stem = "NIA_SL_WORD1501_REAL01_D"
    clip, kept_indices, meta = preprocess_stem(stem, step=5, margin_px=40, img_w=1920, img_h=1080, normalise_imagenet=True)  # your signature

    clip = clip.to(device)                 # (T,3,224,224)

    with torch.no_grad():
        feats = backbone(clip)             # (T,512)

    print(feats.shape)  # should be (T, 512)

if __name__ == "__main__":
    main()
