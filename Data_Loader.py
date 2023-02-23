##IMPORT 

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

class KVasir_dataset(Dataset):
    def __init__(self,train_path,mask_path,image_dir_list,mask_dir_list,transforms=None): #
        super().__init__()
        self.train_path      = train_path
        self.image_dir_list  = image_dir_list
        self.mask_path       = mask_path
        self.mask_dir_list   = mask_dir_list
        self.center_crop     = transforms

    def __len__(self):
        return len(self.image_dir_list)
    
    def __getitem__(self,index):        
        #for idx in range(len(self.image_dir_list)):
            #if 'jpg' in self.image_dir_list:
            image_dir = os.path.join(self.train_path,self.image_dir_list[index])
            image=Image.open(image_dir)
            image=np.array(image,dtype=float)
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image)
            image = self.center_crop(image)
            #if self.transforms is not None:
             #   image = self.transforms(image)
            #self.images.append(image)
        
        #for idx in range(len(self.image_dir_list)):
            #if 'jpg' in self.image_dir_list:
            mask_dir = os.path.join(self.mask_path,self.mask_dir_list[index])
            mask=Image.open(mask_dir)
            mask=np.array(mask,dtype=float)
            mask = np.transpose(mask, (2, 0, 1))
            mask = torch.from_numpy(mask)
            mask = self.center_crop(mask)
            #if self.transforms is not None:
            #    mask = self.transforms(mask)
            #self.masks.append(mask)
            return image, mask
'''

        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = image/255.0 ## (512, 512, 3)
        image = np.transpose(image, (2, 0, 1))  ## (3, 512, 512)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
   ''' 

   
def main():

    train_im_path = "data/images"
    images_dir_list = os.listdir(train_im_path) 
    train_mask_path = "data/masks"
    mask_dir_list = os.listdir(train_mask_path) 
    transformations = transforms.CenterCrop(400)

    data=KVasir_dataset(train_im_path,train_mask_path,images_dir_list,mask_dir_list,transformations)

    train_loader = DataLoader(
        dataset=data,
        batch_size=2,
        shuffle=False,
        num_workers=2 )

    a=data[0]
    b=a[0]

    image,label = next(iter(train_loader))
    print(image.shape)
    print(label.shape)
    
    im=image[0]
    label=label[0]

    fig = plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(b[0].numpy())
    plt.subplot(2,2,2)
    plt.imshow(im[0].numpy())
    plt.subplot(2,2,3)
    plt.imshow(label[0].numpy())
    plt.show()

if __name__ == '__main__':
    
    main()

