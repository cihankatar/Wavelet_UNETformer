import torch
import torch.nn as nn

class Dice_CE_Loss():
    def __init__(self,inputs,targets):

        self.batch,self.h,self.w,self.n_class = inputs.shape
        
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.softmax       = nn.Softmax(dim=-1)

        self.input         = inputs.reshape(self.batch * self.h * self.w ,self.n_class)
        self.target        = targets.reshape(self.batch * self.h * self.w ,self.n_class)

    def Dice_Loss(self):
        smooth=1
        intersection = (self.input * self.target).sum()
        dice_loss    = 1- (2.*intersection + smooth )/(self.input.sum() + self.target.sum() + smooth)
        return dice_loss
 
    def CE_loss(self):
        return self.cross_entropy(self.input, self.target) 
    
    def softmax_manuel(self,input):
        return (torch.exp(input) / torch.sum(torch.exp(input), dim=1, keepdim=True))

    def CE_loss_manuel(self):
        return torch.mean(-torch.sum(torch.log(self.softmax_manuel(self.input)) * (self.target), dim=1, keepdim=True))
    
    def Dice_CE_loss(self):
        return self.Dice_Loss() + self.CE_loss()