import torch
import clip
import numpy as np

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class VClip:
    def __init__(self,clip_model="ViT-B/32",device=device):
        
        model, _ = clip.load(clip_model, device=device)
        self.model=model
        self.device=device
        
    def getVClip(self,labels):
        for i,label in enumerate(labels):
            v = self._getV(label)
            if i==0:
                v_clip = v
            else:
                v_clip = np.concatenate([v_clip,v])
        return v_clip
    
    def _getV(self,label):
        target = clip.tokenize([f'A phot of {label} object']).to(self.device)
        origin = clip.tokenize(['A phot of object']).to(self.device)
        v = self.model.encode_text(target) - self.model.encode_text(origin)
        return (v / torch.norm(v)).type(torch.float16).cpu().detach().numpy()
