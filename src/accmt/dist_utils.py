import torch
import torch.nn.functional as F

def pad_to(tensor: torch.Tensor, maximum: int) -> torch.Tensor:
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

def gather_and_drop_duplicates(tensor: torch.Tensor, maximum: int, accelerator) -> torch.Tensor:
    tensor, padding_mask = pad_to(tensor, maximum)
    
    # gather operations
    tensor = accelerator.gather(tensor)
    padding_mask = accelerator.gather(padding_mask)
    
    tensor = drop_duplicates(tensor, padding_mask)
    
    return tensor
