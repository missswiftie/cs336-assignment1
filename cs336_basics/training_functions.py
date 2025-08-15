import torch
import torch.nn as nn
import numpy as np
import numpy.typing as npt
from typing import BinaryIO,IO
import torch.optim as optim
from os import PathLike


def get_batch(dataset:npt.NDArray,batch_size:int,context_length:int,device:str)->tuple[torch.Tensor,torch.Tensor]:
    max_start_idx=len(dataset)-context_length
    start_indices=torch.randint(0,max_start_idx,(batch_size,),device=device)
    dataset_tensor=torch.as_tensor(dataset,dtype=torch.long)
    inputs_indices=start_indices.unsqueeze(1)+torch.arange(0,context_length,device=device)
    label_indices=inputs_indices+1
    inputs=dataset_tensor[inputs_indices.cpu()]
    labels=dataset_tensor[label_indices.cpu()]
    inputs=inputs.to(device)
    labels=labels.to(device)
    return inputs,labels

def save_checkpointing(model:nn.Module,optimizer:optim.Optimizer,iteration:int,
                       out:str|PathLike|BinaryIO|IO[bytes]):
    checkpoint={
        "iteration":iteration,
        "model_state_dict":model.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        "model_type":type(model).__name__
    }
    if isinstance(out,(str,PathLike)):
        torch.save(checkpoint,out)
    else:
        if not hasattr(out,"write") or not callable(out.write):
            raise TypeError("out must be a file path or a binary file-like object with write method")
        if 'b' not in getattr(out,'mode',''):
            raise ValueError("File-like object must be opened in binary mode")
        torch.save(checkpoint,out)
    if hasattr(out, 'flush'):
        try:
            out.flush()
        except Exception as e:
            print(f"Warning: Could not flush file: {e}")

def load_checkpointing(src:str|PathLike|BinaryIO|IO[bytes],model:nn.Module,optimizer:optim.Optimizer):
    try:
        device=next(model.parameters()).device
    except:
        device=torch.device("cpu")
    checkpoint=torch.load(src,map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint["iteration"])
