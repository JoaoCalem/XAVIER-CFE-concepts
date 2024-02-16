import os
import gdown
import clip
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from pathlib import Path
import pandas as pd

from xavierconcepts.ClipClassifier import ClipClassifier
from xavierconcepts.VClip import VClip
from xavierconcepts.ConceptClip import ConceptClip

path = Path(__file__).parent

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

clip_model="ViT-B/32"

_, preprocess = clip.load(clip_model, device=device)

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=lambda x:preprocess(x).type(torch.float16),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=lambda x:preprocess(x).type(torch.float16),
)

batch_size = 32
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

x = test_data[1][0].to(device)

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

def getClassifier():
    
    print('Loading CLIP Classification Model')
    if not os.path.exists(os.path.join(path,'models')):
        os.mkdir(os.path.join(path,'models'))
    if not os.path.exists(os.path.join(path,'models','model.pth')):
        print('Downloading Classifier from Google Drive')
        url = 'https://drive.google.com/file/d/1-zF8AKHzgCFpNopBZ1w9T3QxhxHqtXV2/'
        gdown.download(id='1-zF8AKHzgCFpNopBZ1w9T3QxhxHqtXV2', output=os.path.join(path,'models','model.pth'))
    model = ClipClassifier().to(device)
    model.load_state_dict(torch.load(os.path.join(path,'models','model.pth')))
    return model

def getVClip(name='label', clip_model="ViT-B/32"):
    labels = pd.read_csv(os.path.join(path,'concepts',f'{name}.csv')).name.map(lambda x: x.split('-')[0])
    vclipfile = os.path.join(path,'concepts',f'v_clip_{name}.pt')
    if not os.path.exists(vclipfile):
        print('Calculating VCLIP')
        v_clip = VClip(device=device).getVClip(labels)
        torch.save(v_clip, vclipfile)
    else:
        print('Loading VCLip')
        v_clip = torch.load(vclipfile)

    return labels, v_clip

def getConcepts(x=x, target=9, classifier = getClassifier(), classes=classes,
                v_clip_name = 'label', alpha=0.1, device=device):
    
    embedding = classifier.clip(x.unsqueeze(0))
    
    labels, v_clip = getVClip(v_clip_name)
    
    model=ConceptClip(embedding, v_clip, classifier).to(device)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(),lr=0.001)

    y_target = target+1
    min_epochs = 10000
    epoch = 0
    
    print("Current Class:", classes[classifier(x.unsqueeze(0))[0].argmax(0)])
    print("Counterfactual Class:", classes[target])
    print('Optimising \'w\'')
    while y_target!=target or min_epochs>epoch:
        input = torch.tensor([1.0]).unsqueeze(0).to(device)
        pred_y = model(input)
        all_linear_params = torch.cat([p.view(-1) for p in model.w.parameters()])
        w = model.w(input)

        loss = loss_fn(pred_y, torch.tensor([target]).to(device)) \
            + alpha*torch.norm(all_linear_params,1) \
            + alpha*torch.norm(all_linear_params,2)

        optim.zero_grad()
        loss.backward()
        optim.step()

        y_target = pred_y.argmax().item()

        epoch += 1

        if epoch%(min_epochs/10) == min_epochs/10-1:
            print(epoch, loss.item(), y_target)
            
    final_w = model.w(input).cpu().detach().numpy()[0]
    return pd.DataFrame({'weight':final_w,'name':labels}).sort_values('weight',ascending=False)
