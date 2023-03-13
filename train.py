
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.optim import Adam
from tqdm import tqdm, trange
import torchvision.transforms as transforms
import torch.nn as nn 
from one_hot_encode import one_hot,label_encode
from data_loader import loader
from Model import build_unet
from Loss import Dice_CE_Loss

def main():
    n_classes   = 1
    batch_size  = 10
    num_workers = 2
    epochs      = 30
    l_r         = 0.001

    train_loader,test_loader = loader(batch_size,num_workers,shuffle=True)
    
    model     = build_unet(n_classes)
    optimizer = Adam(model.parameters(), lr=l_r)
    loss_function      = Dice_CE_Loss()
    loss_sum = 0.0
    
    for epoch in trange(epochs, desc="Training"):
        epoch_loss = []
        loss = 0.0
        loss_sum = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):

            images,labels   = batch                
            model_output    = model(images)

            if n_classes == 1:
                
                model_output     = model_output.squeeze()
                #label           = label_encode(labels)
                #loss            = Dice_Loss(model_output, labels)
                #loss            = BCE_loss(model_output, labels)
                loss             = loss_function.Dice_BCE_Loss(model_output, labels)

            else:
                model_output    = torch.transpose(model_output,1,3) 
                targets_m     = one_hot(labels,n_classes)
                loss_m        = loss_function.CE_loss_manuel(model_output, targets_m)

                targets_f     = label_encode(labels) 
                loss          = loss_function.CE_loss(model_output, targets_f)


            loss     += loss.detach().item() / len(train_loader)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum = loss + loss_sum

            print(f"Epoch {epoch + 1}/{epochs} Batch loss: {loss} loss_sum {loss_sum}")

        epoch_loss.append(loss_sum.detach().numpy())
'''     
        DEBUG --> Copy and paste below lines in order to see current output --<  DEBUG

loss_sum = loss.detach().numpy()
loss_sum = loss_sum.astype(int)
loss_sum1 = loss_sum.tolist()
loss_sum1.append()


sigmoid_f  = nn.Sigmoid()
im_test    = np.array(images[0]*255,dtype=int)
im_test    = np.transpose(im_test, (2,1,0))
label_test = np.array(labels[0],dtype=int)
label_test = np.transpose(label_test)
prediction = sigmoid_f(model_output[0])

if n_classes>1:
    prediction = torch.argmax(prediction,dim=2)    #for multiclass_segmentation

else:
    prediction   = prediction.squeeze()
    
prediction   = prediction.detach().numpy()
prediction   = prediction > 0.5
prediction   = np.array(prediction, dtype=np.uint8)
prediction   = np.transpose(prediction)

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(im_test)
plt.subplot(1, 3, 2)
plt.imshow(label_test,cmap='gray')
plt.subplot(1, 3, 3)
plt.imshow(prediction,cmap='gray')

'''

if __name__ == "__main__":
   main()