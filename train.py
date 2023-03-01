
import numpy as np
import matplotlib.pyplot as plt
from data_loader import loader
from torch.optim import Adam
from tqdm import tqdm, trange
import torch
import torchvision
import torchvision.transforms as transforms
from Model import build_unet

def one_hot_encode(target, n_classes):
    channel,h,w=target.shape
    label_zero = 0
    label_one  = 255

    if channel>1 :

        target_rgbonehot = torch.zeros((channel,h,w,n_classes))    
        target_onehot    = torch.zeros((h,w,n_classes))  
        
        for idx,target_one_channel in enumerate(target):

            for i in range(h):
                for j in range (w):
                    
                    if target_one_channel[j,i]==label_zero:
                        target_onehot[j,i] = torch.tensor([1,0])
                    else:
                        target_onehot[j,i] = torch.tensor([0,1])
                    
            target_rgbonehot[idx,:,:,:]=target_onehot

    return target_rgbonehot 

def main():
    n_classes = 1
    batch_size   = 2
    num_workers  = 2
    epochs        = 10
    l_r = 0.005

    train_loader,test_loader = loader(batch_size,num_workers,shuffle=True)


    images,labels  = next(iter(train_loader))
    
    x = torch.randn((2, 3, 512, 512))
    
    model = build_unet()
    
    y = model(images)


    channel,h,w    = labels.shape[1:]

    if channel>1:

        target_masks_one_hot   = torch.zeros(batch_size,channel,h,w,n_classes)

        for idx, label in enumerate(labels):
            target_mask  = one_hot_encode(label,n_classes)
            target_masks_one_hot[idx] = target_mask
    
    else:
        labels=labels/255.0

    im  = np.array(images[0],dtype=int)
    im  = np.transpose(im, (2, 1, 0))
    lab = np.array(labels[0],dtype=int)
    lab = np.transpose(lab, (2, 1, 0))

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im)
    plt.subplot(1, 2, 2)
    plt.imshow(lab)
    
    plt.imshow(lab)

'''
    model     = UNET((1, 28, 28),n_heads=2, output_dim=10,mlp_layer_size=8)    
    optimizer = Adam(model.parameters(), lr=l_r)
    criterion = Dice_BCE_Loss()
    
    for epoch in trange(epochs, desc="Training"):
        train_loss = 0.0

        for batch in tqdm(loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            
            image, masks    = batch
            predicted_masks  = model(x)
            loss = criterion(predicted_masks, masks)
            train_loss += loss.detach().cpu().item() / len(train_loader)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs} loss: {train_loss}")
'''


if __name__ == "__main__":
   main()