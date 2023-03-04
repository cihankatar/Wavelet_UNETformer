##IMPORT 
import os
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class KVasir_dataset(Dataset):
    def __init__(self,train_path,mask_path,image_dir_list,mask_dir_list,transforms=None): #
        super().__init__()
        self.train_path      = train_path
        self.image_dir_list  = image_dir_list
        self.mask_path       = mask_path
        self.mask_dir_list   = mask_dir_list
        self.tr    = transforms

    def __len__(self):
        return len(self.image_dir_list)
    
    def __getitem__(self,index):        
            #if 'jpg' in self.image_dir_list:
            image_dir = os.path.join(self.train_path,self.image_dir_list[index])
            image = Image.open(image_dir)
            image = np.array(image,dtype=float)
            image = image.astype(np.float32)
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image)
            image = self.tr(image)
            image = image/255.0 
            #if self.transforms is not None:
             #   image = self.transforms(image)
        
            #if 'jpg' in self.image_dir_list:
            mask_dir = os.path.join(self.mask_path,self.mask_dir_list[index])
            mask = Image.open(mask_dir)
            #.convert('L')
            mask = np.array(mask,dtype=float)
            mask = np.transpose(mask, (2, 0, 1))
            mask = torch.from_numpy(mask)
            mask = self.tr(mask)
            mask = mask/255.0 
            mask = mask[0,:,:]
            #if self.transforms is not None:
            #    mask = self.transforms(mask)

            return image, mask
'''

        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = image/255.0 ## (512, 512, 3)
        image = np.transpose(image, (2, 0, 1))  ## (3, 512, 512)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
''' 