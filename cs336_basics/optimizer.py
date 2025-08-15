import torch
import torch.optim as optim
from collections.abc import Callable, Iterable
from typing import Optional
import math

class MyAdamW(optim.Optimizer):
    def __init__(self,params,lr=0.001,betas=(0.9,0.99),eps=1e-8,weight_decay=0.01):
        default=dict(lr=lr,betas=betas,eps=eps,weight_decay=weight_decay)
        super(MyAdamW,self).__init__(params,default)

    def step(self,closure:Optional[Callable]=None):
        loss=None if closure is None else closure()
        for group in self.param_groups:
            lr=group['lr']
            beta1,beta2=group["betas"]
            eps=group["eps"]
            weight_decay=group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state=self.state[p]
                grad=p.grad.data
                if len(state)==0:
                    state["num_steps"]=0
                    state["running_momentum"]=torch.zeros_like(p,memory_format=torch.preserve_format)
                    state["running_v"]=torch.zeros_like(p,memory_format=torch.preserve_format)
                state["num_steps"]+=1
                state["running_momentum"]=beta1*state["running_momentum"]+(1-beta1)*grad
                state["running_v"]=beta2*state["running_v"]+(1-beta2)*(grad**2)
                new_lr=lr*(math.sqrt(1-beta2**state["num_steps"])/(1-beta1**state["num_steps"]))
                p.data.add_(state["running_momentum"]/(state["running_v"].sqrt()+eps),alpha=-new_lr)
                p.data.add_(p.data,alpha=-weight_decay*lr)
        return loss

def lr_cos_annealing(it:int,max_learning_rate:float,min_learning_rate:float,
                     warmup_iters:int,cosine_cycle_iters:int):
    if it<warmup_iters:
        return it*max_learning_rate/warmup_iters
    elif it>cosine_cycle_iters:
        return min_learning_rate
    else:
        return min_learning_rate+0.5*(max_learning_rate-min_learning_rate)*(1+math.cos((it-warmup_iters)/(cosine_cycle_iters-warmup_iters)*math.pi))
    
def gradient_clipping(parameters:Iterable[torch.nn.Parameter],max_l2_norm:float):
    total_norm=torch.zeros(1,device=parameters[0].grad.device)
    for param in parameters:
        if param.grad is None:
            continue
        grad=param.grad.data
        total_norm=total_norm+(grad**2).sum()
    total_norm_sqrt=torch.sqrt(total_norm)
    if total_norm_sqrt>max_l2_norm:
        factor=max_l2_norm/(total_norm_sqrt+1e-6)
        for param in parameters:
            if param.grad is None:
                continue
            param.grad.data=param.grad.data*factor