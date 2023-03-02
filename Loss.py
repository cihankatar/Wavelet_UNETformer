import torch
import torch.nn as nn
import torch.nn.functional as F

class Dice_CE_Loss():
    def __init__(self,input,target):

        self.cross_entropy = F.cross_entropy()
        self.NLL           = F.nll_loss()
        self.softmax       = nn.Softmax(dim=-1)
        self.input         = input.flatten()
        self.target        = target.flatten()
    
    def Dice_Loss(self, input, target):

        intersection = (input * target).sum()
        dice_loss    = 1- (2.*intersection )/(input.sum() + target.sum())

        return dice_loss
 

    def CE_loss(self):
        return self.cross_entropy(self.input, self.target, reduction='mean') 
    

    def softmax_manuel(self,input):
        return (torch.exp(input.t()) / torch.sum(torch.exp(input), dim=1)).t()
    

    def CE_loss_manuel(self):
        return -torch.sum(torch.log(self.softmax_manuel(self.input)) * (self.target), dim=1)

    def NLL_Loss_F(self):
        
        log_input_softmax = torch.log(self.softmax(self.input))
        NLL_Loss         = self.NLL(log_input_softmax, self.target, reduction='mean')   
        return NLL_Loss
    
    def Dice_CE_loss(self):
        return self.Dice_Loss + self.CE_loss_F
         

    
