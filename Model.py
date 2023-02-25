import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from Custom_Dataset import KVasir_dataset
from Test_Train_Split import split_main


def main():
    train_im_path = "train/images"
    train_mask_path = "train/masks"
    test_im_path = "test/images"
    test_mask_path = "test/masks"

    if not os.path.exists(train_im_path):
        split_main()

    train_im_dir_list = os.listdir(train_im_path) 
    train_mask_dir_list = os.listdir(train_mask_path) 
    test_im_dir_list = os.listdir(train_mask_path) 
    test_mask_dir_list = os.listdir(train_im_path) 

    transformations = transforms.CenterCrop(500)

    data_train = KVasir_dataset(train_im_path,train_mask_path,train_im_dir_list,train_mask_dir_list,transformations)
    data_test  = KVasir_dataset(test_im_path,test_mask_path,test_im_dir_list,test_mask_dir_list,transformations)

    train_loader = DataLoader(
        dataset=data_train,
        batch_size=20,
        shuffle=False,
        num_workers=2 )

    image,label = next(iter(train_loader))

    im=np.array(image[0],dtype=int)
    im = np.transpose(image[0], (2, 1, 0))
    label = np.array(label[0],dtype=int)
    label = np.transpose(label, (2, 1, 0))

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im)
    plt.subplot(1, 2, 2)
    plt.imshow(label)
    plt.show()

if __name__ == "__main__":
    main()