import os
import torch
import numpy as np
from PIL import Image
import torchvision.utils
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import random_split
import cv2

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

create_dir("train/images")
create_dir("train/masks")
create_dir("test/images")
create_dir("test/masks")

im_path = "/Users/cihankatar/Desktop/Github_Repo/Vision_Transformer/Unet/data/images"
images_dir_list = sorted(os.listdir(im_path)) 
mask_path = "/Users/cihankatar/Desktop/Github_Repo/Vision_Transformer/Unet/data/masks"
mask_dir_list = sorted(os.listdir(mask_path))
data_path=[im_path,mask_path,im_path,mask_path]

split_ratio=[int(len(images_dir_list)*0.8),int(len(images_dir_list)*0.2)]

train_idx,test_idx     = random_split(images_dir_list, split_ratio, generator=torch.Generator().manual_seed(42))
train_masks,test_masks = random_split(mask_dir_list, split_ratio, generator=torch.Generator().manual_seed(42))

im_train_list=[images_dir_list[i] for i in train_idx.indices]
im_test_list=[images_dir_list[i] for i in test_idx.indices]

masks_train_list=[mask_dir_list[i] for i in train_masks.indices]
masks_test_list=[mask_dir_list[i] for i in test_masks.indices]

all_list =[im_train_list,masks_train_list,im_test_list,masks_test_list]

im_train_path = '/Users/cihankatar/Desktop/Github_Repo/Vision_Transformer/Unet/train/images'
masks_train_path = '/Users/cihankatar/Desktop/Github_Repo/Vision_Transformer/Unet/train/masks'
im_test_path = '/Users/cihankatar/Desktop/Github_Repo/Vision_Transformer/Unet/test/images'
masks_test_path = '/Users/cihankatar/Desktop/Github_Repo/Vision_Transformer/Unet/test/masks'

all_path=[im_train_path,masks_train_path,im_test_path,masks_test_path]

for list, path, data_path in zip(all_list, all_path, data_path):
    os.chdir(path)
    for idx in range(len(list)):

            img_dir = os.path.join(data_path,list[idx])
            img=Image.open(img_dir)
            img=np.array(img,dtype=float)
            img = img[:, :, [2, 1, 0]]
            
            cv2.imwrite(im_train_list[idx], img)

print(len(im_train_list),len(im_test_list))   
print(len(masks_train_list),len(masks_test_list))   
