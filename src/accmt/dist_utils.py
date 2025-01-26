# Copyright 2025 ghanvert. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Optional, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .utility import WORLD_SIZE


def pad_to(tensor: torch.Tensor, maximum: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad on first dimension."""
    target_rows = maximum

    current_rows, _ = tensor.size()
    rows_to_add = target_rows - current_rows

    if rows_to_add > 0:
        padded_tensor = F.pad(tensor, (0, 0, 0, rows_to_add), mode="constant", value=0)
        padded_tensor[-rows_to_add:] = tensor[-1]
    else:
        padded_tensor = tensor[:target_rows]

    padded_mask = torch.tensor(
        [i < len(tensor) for i in range(len(padded_tensor))], dtype=torch.long, device=padded_tensor.device
    )
    return padded_tensor, padded_mask


def drop_duplicates(tensor: torch.Tensor, padded_mask: torch.Tensor = None) -> torch.Tensor:
    return tensor[padded_mask.bool()]


def gather(tensor: torch.Tensor, num_processes: int = None) -> torch.Tensor:
    if WORLD_SIZE == 1:
        return tensor

    if num_processes is None:
        num_processes = WORLD_SIZE

    tensor_ = tensor.clone()  # TODO does this still apply??
    collected = [torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device) for _ in range(num_processes)]
    dist.all_gather(collected, tensor_)
    collected = torch.cat(collected)

    return collected


def gather_object(obj: Any, num_processes: int = None) -> list[Any]:
    if WORLD_SIZE == 1:
        return obj

    if num_processes is None:
        num_processes = WORLD_SIZE

    collected = [None] * num_processes
    dist.all_gather_object(collected, obj)

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


def unique_gather(obj: Union[torch.Tensor, Any], remainder: int = 0) -> Union[torch.Tensor, Any]:
    obj = gather(obj) if isinstance(obj, torch.Tensor) else gather_object(obj)

    if remainder == 0:
        return obj

    true_size = len(obj) - remainder
    return obj[:true_size]
