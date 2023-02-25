import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from Custom_Dataset import KVasir_dataset

train_im_path = "train/images"
train_dir_list = os.listdir(train_im_path) 
train_mask_path = "train/masks"
mask_dir_list = os.listdir(train_mask_path) 

test_im_path = "train/images"
test_dir_list = os.listdir(train_im_path) 
test_mask_path = "train/masks"
test_dir_list = os.listdir(train_mask_path) 

transformations = transforms.CenterCrop(500)

data_train = KVasir_dataset(train_im_path,train_mask_path,train_dir_list,mask_dir_list,transformations)
data_test  = KVasir_dataset(test_im_path,test_mask_path,test_dir_list,test_dir_list,transformations)

train_loader = DataLoader(
    dataset=data_train,
    batch_size=20,
    shuffle=False,
    num_workers=2 )

test_loader = DataLoader(
    dataset=data_test,
    batch_size=20,
    shuffle=False,
    num_workers=2 )


a=data_train[0]
b=a[0]

image,label = next(iter(train_loader))

im=image[0]
label=label[0]

im=np.array(im,dtype=int)
im = np.transpose(im, (2, 1, 0))

plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(b[0])
plt.subplot(2,2,2)
plt.imshow(im)
plt.subplot(2,2,3)
plt.imshow(label[0])
plt.show()

