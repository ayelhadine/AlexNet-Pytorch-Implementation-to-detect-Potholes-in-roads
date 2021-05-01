import numpy as np
import os
import torch
from torch.utils.data import Dataset
from skimage import io


class PotholeDataset(Dataset):
  def __init__(self,images_path,df,transform=None):
    super().__init__()
    self.images_path=images_path
    self.df=df
    self.transform=transform
  def __len__(self):
    return len(self.df)
  
  def __getitem__(self,idx):
    if torch.is_tensor(idx):
      idx=idx.tolist()
    img_name=os.path.join(self.images_path,self.df.iloc[idx,0])
    img_name=img_name+".JPG"
    img=io.imread(img_name)
    img=img.astype(np.uint8)
    #img=np.transpose(img,(2,0,1))

    targets=self.df.iloc[idx,1]


    if self.transform:
      img=self.transform(img)

    sample={"image":img,"targets":targets}
    return sample