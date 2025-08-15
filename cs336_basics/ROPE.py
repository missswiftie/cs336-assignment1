import torch 
import torch.nn as nn
from einops import rearrange

class ROPE(nn.Module):
    def __init__(self,theta:float,d_k:int,max_seq_len:int,device:torch.device|None=None):
        super(ROPE,self).__init__()
        self.theta=theta
        self.d_k=d_k
        self.max_seq_len=max_seq_len
        self.half_dim=d_k//2
        factor=1.0/(theta**(torch.arange(0,self.d_k,2,dtype=torch.float32,device=device)/self.d_k))
        t=torch.arange(max_seq_len,dtype=torch.float32,device=device)
        expi=torch.outer(t,factor)
        sin=torch.sin(expi)
        cos=torch.cos(expi)
        self.register_buffer("sin_cached",sin,persistent=False)
        self.register_buffer("cos_cached",cos,persistent=False)

    def forward(self,x:torch.Tensor,token_positions:torch.Tensor)->torch.Tensor:
        in_dtype=x.dtype
        x=x.to(torch.float32)
        sin=self.sin_cached[token_positions]
        cos=self.cos_cached[token_positions]
        x_reshaped=rearrange(x,"... seq_len (half_dim two)->... seq_len half_dim two",two=2)
        x_as_complex=torch.view_as_complex(x_reshaped)
        rot=torch.complex(cos.unsqueeze(0),sin.unsqueeze(0))
        rot_as_real=torch.view_as_real(rot*x_as_complex)
        result=rearrange(rot_as_real,"... seq_len half_dim two->... seq_len (half_dim two)",two=2)
        return result.to(in_dtype)