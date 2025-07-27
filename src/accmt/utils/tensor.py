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

from typing import Optional, Union

import torch
import torch.nn.functional as F


def pad(tensor: torch.Tensor, to: int, value: Union[float, int]) -> torch.Tensor:
    """
    Pad a tensor to a specified size.

    Args:
        tensor (`torch.Tensor`):
            Tensor to pad.
        to (`int`):
            Size to pad the tensor to.
        value (`Union[float, int]`):
            Value to pad the tensor with.

    Returns:
        `torch.Tensor`: Padded tensor.
    """
    size = tensor.shape[-1]
    if size >= to:
        return tensor

    pad_size = to - size
    tensor = F.pad(tensor, (0, pad_size), value=value)

    return tensor


def drop_duplicates(tensor: torch.Tensor, padded_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Drop duplicates from a tensor with a given mask.

    Args:
        tensor (`torch.Tensor`):
            Tensor to drop duplicates from.
        padded_mask (`torch.Tensor`, *optional*, defaults to `None`):
            Mask to drop duplicates from.

    Returns:
        `torch.Tensor`: Tensor with duplicates dropped.
    """
    if padded_mask is None:
        return tensor

    return tensor[padded_mask.bool()]
