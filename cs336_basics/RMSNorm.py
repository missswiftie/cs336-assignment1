import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self,d_model:int,eps:float=1e-5,
                 device:torch.device|None=None,dtype:torch.dtype|None=None):
        super(RMSNorm,self).__init__()
        self.d_model=d_model
        self.eps=eps
        self.weight=nn.Parameter(torch.ones(d_model))

    def forward(self,x:torch.Tensor)->torch.Tensor:
        in_dtype=x.dtype
        x=x.to(torch.float32)
        denom=torch.sqrt(torch.mean(x**2,dim=-1,keepdim=True)+self.eps)
        result=(x/denom)*self.weight
        return result.to(in_dtype)
