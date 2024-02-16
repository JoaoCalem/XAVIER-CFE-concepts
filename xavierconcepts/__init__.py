import os
import gdown
import torch
from xavierconcepts.ClipClassifier import ClipClassifier

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

def getClassifier():
    if not os.path.exists('models/'):
        os.mkdir('models/')
    if not os.path.exists('models/model.py'):
        url = 'https://drive.google.com/file/d/1-zF8AKHzgCFpNopBZ1w9T3QxhxHqtXV2/view?usp=sharing'
        gdown.download(url, 'models/model.pth')
    model = ClipClassifier().to(device)
    model.load_state_dict(torch.load("models/model.pth"))
    return model