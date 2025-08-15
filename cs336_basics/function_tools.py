import torch
import torch.nn as nn

def softmax(x:torch.Tensor,dim:int=-1)->torch.Tensor:
    if dim<0:
        dim+=x.ndim
    in_dtype=x.dtype
    x=x.to(torch.float32)
    x_max=torch.amax(x,dim=dim,keepdim=True)
    x_norm=x-x_max
    x_exp=torch.exp(x_norm)
    x_sum=torch.sum(x_exp,dim=dim,keepdim=True)
    result=x_exp/x_sum
    return result.to(in_dtype)

def cross_entropy_loss(logits:torch.Tensor,targets:torch.Tensor):
    batch_idx=torch.arange(logits.shape[0],device=logits.device)
    p_scores=logits-logits.logsumexp(dim=-1,keepdim=True)
    scores=p_scores[batch_idx,targets]
    return -torch.mean(scores).to(dtype=logits.dtype)
