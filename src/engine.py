import torch
import pandas as pd
import torchvision.transforms as ts
from torch.utils.data import DataLoader, random_split
from dataset import PotholeDataset
from model import ALEXNET

def evaluate(model, val_loader,device):
    model.eval()
    outputs=[]
    for batch in val_loader:
        images=batch["image"]
        labels=batch["targets"]
        images=images.to(device)
        labels=labels.to(device)
        output =model.validation_step(images,labels)
        outputs.append(output)
    return model.validation_epoch_end(outputs)



def fit(epochs, lr, model, train_loader, val_loader,device,opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        for batch in train_loader:
            images=batch["image"]
            labels=batch["targets"]
            images=images.to(device)
            labels=labels.to(device)
            loss = model.training_step(images,labels)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader,device)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

if __name__=="__main__":
    device = get_default_device()
    images_path="./input/all_data/"
    train_df=pd.read_csv("./input/train_ids_labels.csv")
    stats=((0.5,0.5,0.5),(0.5,0.5,0.5))
    tfms=ts.Compose([ts.ToTensor(),
                     ts.CenterCrop((120,800)),
                     ts.Resize((227,227)),
                     ts.Normalize(*stats)])
    dataset=PotholeDataset(images_path,train_df,transform=tfms)
    
    batch_size=64
    train_ds,val_ds=random_split(dataset,[len(dataset)-500,500])
    train_loader=DataLoader(train_ds,batch_size,shuffle=True,num_workers=4,pin_memory=True)
    val_loader=DataLoader(val_ds,batch_size*2,num_workers=4,pin_memory=True)
    
    model=ALEXNET(2)
    to_device(model,device)
    run1=fit(10,0.001,model,train_loader,val_loader,device,opt_func=torch.optim.Adam)
    
    
