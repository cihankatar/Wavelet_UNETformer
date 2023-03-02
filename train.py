
import numpy as np
import matplotlib.pyplot as plt
from data_loader import loader
from torch.optim import Adam
from tqdm import tqdm, trange
import torch
import torchvision
import torchvision.transforms as transforms
from Model import build_unet
import torch.nn as nn 
from Loss import Dice_CE_Loss
from one_hot_encode import one_hot


def main():
    n_classes   = 2
    batch_size  = 2
    num_workers = 2
    epochs      = 10
    l_r         = 0.005

    train_loader,test_loader = loader(batch_size,num_workers,shuffle=True)
    
    model = build_unet()
    optimizer = Adam(model.parameters(), lr=l_r)

    for batch in train_loader:

        images,labels  = batch        
        model_output   = model(images)
        targets        = one_hot(labels,n_classes)
        inputs         = torch.transpose(model_output,3,1)

        loss           = Dice_CE_Loss(inputs,targets)
        #CE_loss        = loss.CE_loss()
        #CE_loss_manuel = loss.CE_loss_manuel()
        #dice_loss      = loss.Dice_Loss()
        dice_ce_loss   = loss.Dice_CE_loss()

        im_test    = np.array(images[0],dtype=int)
        im_test    = np.transpose(im_test, (2, 1, 0))
        label_test = np.array(labels[0],dtype=int)
        label_test = np.transpose(label_test)

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(im_test)
        plt.subplot(1, 2, 2)
        plt.imshow(label_test)
        plt.imshow(label_test)
        
        break
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