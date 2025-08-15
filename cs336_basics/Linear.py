import numpy as np
import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self,in_features:int,out_features:int,
                 device: torch.device | None = None, dtype: torch.dtype | None = None):
        super(Linear,self).__init__()
        self.in_features=in_features
        self.out_features=out_features
        self.device=device
        self.dtype=dtype
        w_init=self.init_weight(in_features,out_features,device,dtype)
        self.weight=nn.Parameter(w_init)

    def init_weight(self,in_features:int,out_features:int,device=None,dtype=None):
        weight=torch.empty((in_features,out_features),device=device,dtype=dtype)
        mean=0.0
        std=np.sqrt(2.0/(in_features+out_features))
        nn.init.trunc_normal_(weight,mean,std,-3.0*std,3.0*std)
        return weight
    
    def forward(self,x:torch.Tensor):
        result=torch.matmul(x,self.weight)
        return result