
import numpy as np
import matplotlib.pyplot as plt
from data_loader import loader

def main():

    batch_size   = 5
    num_workers  = 2
    
    train_loader = loader(batch_size,num_workers,shuffle=True)
    image,label  = next(iter(train_loader))

    im  = np.array(image[0],dtype=int)
    im  = np.transpose(im, (2, 1, 0))
    lab = np.array(label[0],dtype=int)
    lab = np.transpose(lab, (2, 1, 0))

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im)
    plt.subplot(1, 2, 2)
    plt.imshow(lab)
    plt.show()

if __name__ == "__main__":
   main()