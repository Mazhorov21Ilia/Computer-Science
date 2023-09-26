import torch
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from tqdm import tqdm




class Dataset2class(torch.utils.data.Dataset):
  def __init__(self, path_dir1:str, path_dir2:str):
    super().__init__()

    self.path_dir1 = path_dir1
    self.path_dir2 = path_dir2

    self.dir1_list = sorted(os.listdir(path_dir1))
    self.dir2_list = sorted(os.listdir(path_dir2))

  def __len__(self):
    return len(self.dir1_list) + len(self.dir2_list)

  def __getitem__(self, idx):

    if idx <= len(self.dir1_list):
      class_id = 0
      img_path = os.path.join(self.path_dir1, self.dir1_list[idx])
    else:
      class_id = 1
      img_path = os.path.join(self.path_dir2, self.dir2_list[idx])


    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img/255.0
    
    img = cv2.resize(img, (85, 85), interpolation=cv2.INTER_AREA)

    img = img.transpose((2, 0, 1))

    t_img = torch.from_numpy(img)

    t_class_id = torch.tensor(class_id)

    return img

    return {'img': t_img,'label': t_class_id}



train_chih_path = '/content/train/chihyahua'
train_muff_path = '/content/train/muffin'
test_chih_path = '/content/test/chihyahua'
test_muff_path = '/content/test/muffin'

train_ds_chihvsmaff = Dataset2class(train_chih_path, train_muff_path)
test_ds_chihvsmaff = Dataset2class(test_chih_path, test_muff_path)



#DataLoader



batch_size = 16

train_loader = torch.utils.data.DataLoader(
    train_ds_chihvsmaff, shuffle=True, 
    batch_size=batch_size, num_workers=1, drop_last=True
)
test_loader = torch.utils.data.DataLoader(
    test_ds_chihvsmaff, shuffle=True, 
    batch_size=batch_size, num_workers=1, drop_last=False
)
