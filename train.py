
import numpy as np
import matplotlib.pyplot as plt
from data_loader import loader

def main():

    train_loader = loader()
    image,label  = next(iter(train_loader))

    im  = np.array(image[0],dtype=int)
    im  = np.transpose(im, (2, 1, 0))
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