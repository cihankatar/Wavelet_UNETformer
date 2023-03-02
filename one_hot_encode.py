
import torch

def one_hot(targets, n_classes):
    batch_size,h,w=targets.shape
    label_zero = 0

    target_onehot = torch.zeros((h,w,n_classes))    
    target_masks_one_hot   = torch.zeros(batch_size,h,w,n_classes)

    for idx, label in enumerate(targets):
        for i in range(h):
            for j in range (w):
                if label[j,i]==label_zero:
                    target_onehot[j,i] = torch.tensor([1,0])
                else:
                    target_onehot[j,i] = torch.tensor([0,1])
        
        target_masks_one_hot[idx] = target_onehot

    return target_masks_one_hot 