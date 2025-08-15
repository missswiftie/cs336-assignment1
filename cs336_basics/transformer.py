import torch
import torch.nn as nn
from .attention import Multihead_self_attention_with_rope
from .RMSNorm import RMSNorm
from .SwiGLU import SwiGLU
from .Embedding import Embedding
from .Linear import Linear


class Transformer_block(nn.Module):
    def __init__(self,d_model:int,num_heads:int,max_seq_len:int,d_ff:int,theta:float,
                 device:torch.Tensor|None=None,dtype:torch.dtype|None=None):
        super(Transformer_block,self).__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.max_seq_len=max_seq_len
        self.head_dim=d_model//num_heads
        self.theta=theta
        self.device=device
        self.d_ff=d_ff
        self.attn=Multihead_self_attention_with_rope(
            d_model,num_heads,max_seq_len,theta,device,dtype
            )
        self.ln1=RMSNorm(d_model,0.00001,device,dtype)
        self.ln2=RMSNorm(d_model,0.00001,device,dtype)
        self.ffn=SwiGLU(d_model,d_ff,device,dtype)

    def forward(self,in_features:torch.Tensor,token_positions:torch.Tensor|None=None):
        if token_positions is None:
            token_positions=torch.arange(in_features.shape[1],device=in_features.device).expand(in_features.shape[0],-1)
        y1=self.attn(self.ln1(in_features),token_positions)+in_features
        y2=self.ffn(self.ln2(y1))+y1
        return y2
    
class Transformer(nn.Module):
    def __init__(self,vocab_size:int,context_length:int,d_model:int,
                 num_layers:int,num_heads:int,d_ff:int,theta:float,
                 device:torch.device|None=None,dtype:torch.dtype|None=None):
        super(Transformer,self).__init__()
        self.vocab_size=vocab_size
        self.context_length=context_length
        self.d_model=d_model
        self.num_layers=num_layers
        self.num_heads=num_heads
        self.d_ff=d_ff
        self.theta=theta
        self.dtype=dtype if dtype is not None else torch.float32
        self.device=device
        self.embedding=Embedding(vocab_size,d_model,device,self.dtype)
        self.rmsnorm=RMSNorm(d_model,0.00001,device,self.dtype)
        self.linear=Linear(d_model,vocab_size,device,self.dtype)
        self.transformer_blocks={}
        for i in range(num_layers):
            self.transformer_blocks[f"layer{i+1}"]=Transformer_block(
                d_model,num_heads,context_length,d_ff,theta,device,self.dtype)
    
    def forward(self,in_indices:torch.Tensor)->torch.Tensor:
        input_data=self.embedding(in_indices)
        for i in range(1,self.num_layers+1):
            input_data=self.transformer_blocks[f'layer{i}'](input_data)
        input_data=self.linear(self.rmsnorm(input_data))
        return input_data