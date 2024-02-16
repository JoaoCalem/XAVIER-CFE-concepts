import torch

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class ConceptClip(torch.nn.Module):
    def __init__(self, embedding, v_clip, classifier, device=device):
        super().__init__()
        self.w = torch.nn.Linear(1, v_clip.shape[0])
        self.v_clip = torch.tensor(v_clip).unsqueeze(0).type(torch.float32).to(device)
        self.embedding = embedding
        self.model = classifier

        self.v_clip.requires_grad_(False)
        self.embedding.requires_grad_(False)
        self.model.requires_grad_(False)

        self.softmax = torch.nn.Softmax()

    def forward(self,x):
        return self.model.linear(self.embedding + torch.tensordot(self.w(x),self.v_clip))