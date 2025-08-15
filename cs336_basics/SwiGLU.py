import numpy as np
import torch
import torch.nn as nn
from einops import einsum

def SiLU(x:torch.Tensor)->torch.Tensor:
    in_dtype=x.dtype
    x=x.to(torch.float32)
    return (x*torch.sigmoid(x)).to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self,d_model:int,d_ff:int,
                 device:torch.device|None=None,dtype:torch.dtype|None=None):
        super(SwiGLU,self).__init__()
        self.d_model=d_model
        self.d_ff=d_ff
        w1_init=self.init_weight(d_ff,d_model,device,dtype)
        w3_init=self.init_weight(d_ff,d_model,device,dtype)
        w2_init=self.init_weight(d_model,d_ff,device,dtype)
        self.weight1=nn.Parameter(w1_init)
        self.weight2=nn.Parameter(w2_init)
        self.weight3=nn.Parameter(w3_init)

    def init_weight(self,in_features:int,out_features:int,device=None,dtype=None):
        weight=torch.empty((in_features,out_features),device=device,dtype=dtype)
        mean=0.0
        std=np.sqrt(2.0/(in_features+out_features))
        nn.init.trunc_normal_(weight,mean,std,-3.0*std,3.0*std)
        return weight
    
    def forward(self,x:torch.Tensor)->torch.Tensor:
        in_dtype=x.dtype
        x=x.to(torch.float32)
        w1x=einsum(x,self.weight1,"... d_model,d_ff d_model->... d_ff")
        w3x=einsum(x,self.weight3,"... d_model,d_ff d_model->... d_ff")
        result=einsum(SiLU(w1x)*w3x,self.weight2,"... d_ff,d_model d_ff->... d_model")
        return result.to(in_dtype)