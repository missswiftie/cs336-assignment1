import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self,num_embeddings:int,embedding_dim:int,
                 device:torch.device|None=None,dtype:torch.dtype|None=None):
        super(Embedding,self).__init__()
        self.num_embeddings=num_embeddings
        self.embedding_dim=embedding_dim
        w_init=self.init_weight(self.num_embeddings,self.embedding_dim,device,dtype)
        self.weight=nn.Parameter(w_init)
    
    def init_weight(self,num_embeddings:int,embedding_dim:int,device=None,dtype=None):
        weight=torch.empty((num_embeddings,embedding_dim),device=device,dtype=dtype)
        mean=0.0
        nn.init.trunc_normal_(weight,mean,1.0,-3.0,3.0)
        return weight
    
    def forward(self,token_ids:torch.Tensor)->torch.Tensor:
        return self.weight[token_ids]