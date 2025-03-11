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


def gather_into_single_process(tensor: Optional[torch.Tensor], dst: int = 0, remainder: int = 0) -> torch.Tensor:
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
        if remainder > 0:
            collected = collected[:remainder]
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


class Gatherer:
    def __init__(self):
        from . import accelerator

        self.accelerator = accelerator

    def all_gather_dictionary(self, tensors: dict[Any, torch.Tensor]) -> dict[Any, torch.Tensor]:
        """
        Perform an 'all_gather' operation across all devices for a dictionary of tensors. This
        will remove any possible duplicates and will handle cases where keys are different
        between processes.

        Args:
            tensors (`dict`):
                Dictionary containing tensors.

        Returns:
            `dict`: Gathered dictionary of tensors.
        """
        all_keys = self.accelerator.gather_for_metrics(tensors, use_gather_object=True)  # this will gather only keys

        # IMPORTANT: it is mandatory to have keys sorted
        all_keys = list(sorted(set(all_keys)))
        local_keys = list(sorted(tensors.keys()))

        sample = tensors[local_keys[0]]
        size0 = sample.shape[0]
        device = sample.device
        padding_mask = {}

        for key in all_keys:
            existing_key = True
            if key not in local_keys:
                pad_tensor = torch.zeros(size0, dtype=torch.bool, device=device)
                tensors[key] = torch.zeros_like(sample)
                padding_mask[key] = pad_tensor
                existing_key = False

            global_max_size = torch.tensor(tensors[key].shape[0], dtype=torch.int64, device=device)
            global_max_size = self.accelerator.gather(global_max_size).max().item()

            # verify if tensor is less size than maximum, then pad to max size
            local_size = tensors[key].shape[0]
            if local_size < global_max_size:
                diff = global_max_size - local_size
                _pad = (0, 0, 0, diff) if tensors[key].ndim >= 2 else (0, diff)
                tensors[key] = F.pad(tensors[key], _pad)  # pad to bottom (or right) with 0s

                if existing_key:
                    pad_tensor = torch.ones(tensors[key].shape[0], dtype=torch.bool, device=device)
                    pad_tensor[-diff:] = 0  # apply 0 to the lasts 'diff' elements
                else:
                    pad_tensor = F.pad(pad_tensor, (0, diff))  # pad to right with 0s

                padding_mask[key] = pad_tensor
            elif existing_key:  # if everything is correct, declare key and padding tensor of ones
                padding_mask[key] = torch.ones(global_max_size, dtype=torch.bool, device=device)

        # sort keys again
        padding_mask = {k: padding_mask[k] for k in sorted(padding_mask)}
        tensors = {k: tensors[k] for k in sorted(tensors)}

        # gather
        padding_mask = self.accelerator.gather_for_metrics(padding_mask)
        tensors = self.accelerator.gather_for_metrics(tensors)

        # mask out padding elements
        for k, v in tensors.items():
            tensors[k] = v[padding_mask[k]]

        return tensors
