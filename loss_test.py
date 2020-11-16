import torch
import torch.nn as nn
import numpy as np

def calc_loss(output_pred, ground_truth, n_class = 20, grid_size=13, device="cpu", anchor_box=np.load('./dataset/anchor.npy'):

    batch_size, _, height, width = output_pred.size()
    
    output_pred = output_pred.permute(0,2,3,1).contiguous().view(batch_size, grid_size**2*5, -1)  #(B,H,W,C)
    coord = torch.zeros_like(output_pred[:,:,0:4])
    coord[:,:,0:2] = output_pred[:,:,0:2].nn.Sigmoid()
    coord[:,:,2:4] = output_pred[:,:,2:4].torch.exp()
    confid = output_pred[:,:,4:5].nn.Sigmoid()
    classifi = output[:,:,5:].view(-1,n_class)
    