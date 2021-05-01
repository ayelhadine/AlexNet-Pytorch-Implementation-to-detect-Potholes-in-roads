import torch
import torch.nn as nn
import torch.nn.functional as F

def accuracy(out,labels):
  _,preds=torch.max(out,dim=1)
  return torch.tensor(torch.sum(torch.eq(preds,labels)).item()/len(preds))
  
  
class ImageClassificationBase(nn.Module):
  def training_step(self,images,labels):
 
    out=self(images)
    loss=F.cross_entropy(out,labels)
    return loss
  
  def validation_step(self,images,labels):
    out=self(images)
    loss=F.cross_entropy(out,labels)
    acc=accuracy(out,labels)
    return {"val_loss":loss.detach(),"val_acc":acc}
  
  def validation_epoch_end(self,outputs):
    batch_losses=[x["val_loss"] for x in outputs]
    epoch_loss=torch.stack(batch_losses).mean()
    batch_accs=[x["val_acc"] for x in outputs]
    epoch_acc=torch.stack(batch_accs).mean()
    return {"val_loss":epoch_loss.item(),"val_acc":epoch_acc.item()}
 
  def epoch_end(self,epoch,result):
    print("Epoch [{}]: train_loss {:.4f}, val_loss {:.4f}, val_acc {:.4f}".format(epoch,result["train_loss"],result["val_loss"],result["val_acc"]))
    
    
class ALEXNET(ImageClassificationBase):
  def __init__(self,num_classes):
    super().__init__()
    self.network=nn.Sequential(
        #1 Conv
        nn.Conv2d(3,96,kernel_size=11,stride=4),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3,stride=2),
        #2 Conv
        nn.Conv2d(96,256,kernel_size=5,padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3,stride=2),
        #3 Conv
        nn.Conv2d(256,384,kernel_size=3,padding=1),
        nn.ReLU(inplace=True),
        #4 Conv
        nn.Conv2d(384,384,kernel_size=3,padding=1),
        nn.ReLU(inplace=True),
        #5 Conv
        nn.Conv2d(384,256,kernel_size=3,padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3,stride=2),
        
        nn.Flatten(),
        nn.Dropout(0.5),
        nn.Linear(256*6*6,4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(4096,4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096,num_classes)
 )
  
  def forward(self,xb):
    return self.network(xb)