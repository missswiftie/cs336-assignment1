import torch
import torch.nn as nn
from .function_tools import softmax
import math
from einops import einsum
import numpy as np
from .ROPE import ROPE

class Scaled_dot_product_attention(nn.Module):
    def __init__(self):
        super(Scaled_dot_product_attention,self).__init__()

    def forward(self,
                Q:torch.Tensor,
                K:torch.Tensor,
                V:torch.Tensor,
                mask:torch.Tensor|None=None)->torch.Tensor:
        d_k=math.sqrt(Q.shape[-1])
        qk_score=einsum(Q,K,"... queries d_k,... keys d_k->... queries keys")/d_k
        if mask is not None:
            qk_score=qk_score.masked_fill(~mask,float('-inf'))
        result=torch.matmul(softmax(qk_score,dim=-1),V)
        return result

class Casual_multihead_self_attention(nn.Module):
    def __init__(self,d_model:int,num_heads:int,
                 device:torch.device|None=None,dtype:torch.dtype|None=None):
        super(Casual_multihead_self_attention,self).__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.head_dim=d_model//num_heads
        self.device=device
        self.weight_q=nn.Parameter(self.init_weight(d_model,d_model,device,dtype))
        self.weight_k=nn.Parameter(self.init_weight(d_model,d_model,device,dtype))
        self.weight_v=nn.Parameter(self.init_weight(d_model,d_model,device,dtype))
        self.weight_o=nn.Parameter(self.init_weight(d_model,d_model,device,dtype))

    def init_weight(self,in_features:int,out_features:int,device=None,dtype=None):
        weight=torch.empty((in_features,out_features),device=device,dtype=dtype)
        mean=0.0
        std=np.sqrt(2.0/(in_features+out_features))
        nn.init.trunc_normal_(weight,mean,std,-3.0*std,3.0*std)
        return weight
    
    def forward(self,in_features:torch.Tensor)->torch.Tensor:
        in_dtype=in_features.dtype
        in_features=in_features.to(torch.float32)
        Q=torch.matmul(in_features,self.weight_q.T)#b*N*d_k
        K=torch.matmul(in_features,self.weight_k.T)#b*N*d_k
        V=torch.matmul(in_features,self.weight_v.T)#b*N*d_v
        batch_size,seq_len,_=in_features.shape
        Q_reshaped=Q.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        K_reshaped=K.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        V_reshaped=V.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        scores=torch.matmul(Q_reshaped,K_reshaped.transpose(-2,-1))/(self.head_dim**0.5)
        mask=torch.tril(torch.ones(seq_len,seq_len,dtype=torch.bool,device=self.device))
        attn_scores=scores.masked_fill(mask==0,float('-inf'))
        result=torch.matmul(softmax(attn_scores,dim=-1),V_reshaped).transpose(1,2).contiguous().view(batch_size,seq_len,-1)#b*s*d_model
        out=torch.matmul(result,self.weight_o.T)
        return out.to(in_dtype)
    
class Multihead_self_attention_with_rope(nn.Module):
    def __init__(self,d_model:int,num_heads:int,max_seq_len:int,theta:float,
                 device:torch.Tensor|None=None,dtype:torch.dtype|None=None):
        super(Multihead_self_attention_with_rope,self).__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.max_seq_len=max_seq_len
        self.head_dim=d_model//num_heads
        self.theta=theta
        self.device=device
        self.rope_layer=ROPE(theta,self.head_dim,max_seq_len,device)
        self.weight_q=nn.Parameter(self.init_weight(d_model,d_model,device,dtype))
        self.weight_k=nn.Parameter(self.init_weight(d_model,d_model,device,dtype))
        self.weight_v=nn.Parameter(self.init_weight(d_model,d_model,device,dtype))
        self.weight_o=nn.Parameter(self.init_weight(d_model,d_model,device,dtype))

    def init_weight(self,in_features:int,out_features:int,device=None,dtype=None):
        weight=torch.empty((in_features,out_features),device=device,dtype=dtype)
        mean=0.0
        std=np.sqrt(2.0/(in_features+out_features))
        nn.init.trunc_normal_(weight,mean,std,-3.0*std,3.0*std)
        return weight
    
    def forward(self,in_features:torch.Tensor,token_positions:torch.Tensor)->torch.Tensor:
        in_dtype=in_features.dtype
        in_features=in_features.to(torch.float32)
        Q=torch.matmul(in_features,self.weight_q.T)#b*N*d_k
        K=torch.matmul(in_features,self.weight_k.T)#b*N*d_k
        V=torch.matmul(in_features,self.weight_v.T)#b*N*d_v
        batch_size,seq_len,_=in_features.shape
        Q_reshaped=Q.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        K_reshaped=K.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        V_reshaped=V.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        Q_reshaped=self.rope_layer(Q_reshaped,token_positions)
        K_reshaped=self.rope_layer(K_reshaped,token_positions)
        scores=torch.matmul(Q_reshaped,K_reshaped.transpose(-2,-1))/(self.head_dim**0.5)
        mask=torch.tril(torch.ones(seq_len,seq_len,dtype=torch.bool,device=self.device))
        attn_scores=scores.masked_fill(mask==0,float('-inf'))
        result=torch.matmul(softmax(attn_scores,dim=-1),V_reshaped).transpose(1,2).contiguous().view(batch_size,seq_len,-1)#b*s*d_model
        out=torch.matmul(result,self.weight_o.T)
        return out.to(in_dtype)
