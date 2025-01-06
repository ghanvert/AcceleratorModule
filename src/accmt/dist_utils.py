import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from .utility import RANK, WORLD_SIZE
from typing import Optional

def pad_to(tensor: torch.Tensor, maximum: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad on first dimension."""
    target_rows = maximum

    current_rows, _ = tensor.size()
    rows_to_add = target_rows - current_rows

    if rows_to_add > 0:
        padded_tensor = F.pad(tensor, (0, 0, 0, rows_to_add), mode='constant', value=0)
        padded_tensor[-rows_to_add:] = tensor[-1]
    else:
        padded_tensor = tensor[:target_rows]
        
    padded_mask = torch.tensor([i < len(tensor) for i in range(len(padded_tensor))], dtype=torch.long, device=padded_tensor.device)
    return padded_tensor, padded_mask

def drop_duplicates(tensor: torch.Tensor, padded_mask: torch.Tensor = None) -> torch.Tensor:
    return tensor[padded_mask.bool()]


def gather(tensor: torch.Tensor, num_processes: int = None) -> torch.Tensor:
    if WORLD_SIZE == 1:
        return tensor
    
    tensor_ = tensor.clone()
    collected = [torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device) for _ in range(num_processes)]
    dist.all_gather(collected, tensor_)
    collected = torch.cat(collected)
    
    return collected

def gather_into_single_process(tensor: Optional[torch.Tensor], dst: int = 0) -> torch.Tensor:
    if WORLD_SIZE == 1:
        return tensor

    if tensor is not None and tensor.ndimension() == 0:
        tensor = tensor.unsqueeze(0)
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    collected = [None for _ in range(world_size)]
    dist.gather_object(tensor, collected if rank == dst else None, dst=dst)
    output = None
    if rank == dst:
        rank_device = f"cuda:{dst}"
        tensors = [tensor.to(rank_device) for tensor in collected if tensor is not None]
        if len(tensors) > 0:
            output = torch.cat(tensors)
        
    return output

def gather_and_drop_duplicates(tensor: torch.Tensor, maximum: int) -> torch.Tensor:
    tensor, padding_mask = pad_to(tensor, maximum)
    
    # gather operations
    tensor = gather(tensor)
    padding_mask = gather(padding_mask)
    
    tensor = drop_duplicates(tensor, padding_mask)
    
    return tensor
