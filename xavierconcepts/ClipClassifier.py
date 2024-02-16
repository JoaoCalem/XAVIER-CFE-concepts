import clip
import torch
import torch.nn as nn

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class ClipClassifier(nn.Module):
    def __init__(self, clip_model="ViT-B/32"):
        super().__init__()
        model, _ = clip.load(clip_model, device=device)
        self.dtype = model.dtype
        self.clip = model.visual
        self.clip.requires_grad_(False)
        self.linear = nn.Linear(512, 10)
        self.linear.requires_grad_(True)

    def forward(self,x):
        embedding = self.clip(x)
        logits = self.linear(embedding.type(torch.float32))
        return logits