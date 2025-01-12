# Copyright 2022 ghanvert. All rights reserved.
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

import os
from typing import Union

import numpy as np
import pandas as pd
from typing_extensions import Any

from .utils import divide_list


WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
RANK = int(os.getenv("RANK", 0))
MASTER_PROCESS = RANK == 0
LAST_PROCESS = RANK == WORLD_SIZE - 1


def prepare_list(lst: list, even: bool = False) -> list:
    """
    Get an specific partition of a list per process.

    Args:
        lst (`list`):
            List containing all data.
        even (`bool`, *optional*, defaults to `False`):
            Wether to create even partitions across all processes.

    Returns:
        `list`: Rank-specific list partition.
    """
    if WORLD_SIZE > 1:
        return divide_list(lst, WORLD_SIZE, even=even)[RANK]

    return lst


def prepare_dataframe(df: pd.DataFrame, even: bool = False) -> tuple[pd.DataFrame, int]:
    """
    Get an specific partition of a Pandas DataFrame per process.

    Args:
        df (`pd.DataFrame`):
            Pandas DataFrame.
        even (`bool`, *optional*, defaults to `False`):
            Wether to create even partitions across all processes.

    Returns:
        `list`: Rank-specific Pandas DataFrame partition.
    """
    remainder = 0
    if WORLD_SIZE > 1:
        partition_size, remainder = divmod(len(df), WORLD_SIZE)

        for rank, i in enumerate(range(0, len(df), partition_size + remainder)):
            if rank == RANK:
                df = df.iloc[i : i + partition_size + remainder]
                break

        if even and LAST_PROCESS and remainder != 0:
            last_row = df.iloc[-1:]
            df = pd.concat([df] + [last_row] * (remainder * (WORLD_SIZE - 1)), ignore_index=True)

        remainder = remainder * (WORLD_SIZE - 1)

    return df, remainder


def prepare_array(array: Any, even: bool = False) -> Union[np.ndarray, Any]:
    """
    Get an specific partition of a Pandas DataFrame per process.

    Args:
        array (`Any`):
            Array-like.
        even (`bool`, *optional*, defaults to `False`):
            Wether to create even partitions across all processes.

    Returns:
        `list`: Rank-specific array-like partition.
    """
    if WORLD_SIZE > 1:
        if even:
            part_size = -(-len(array) // WORLD_SIZE)
            total_elements_needed = part_size * WORLD_SIZE
            padding_value = array[-1]
            array = np.pad(
                array, (0, total_elements_needed - len(array)), mode="constant", constant_values=padding_value
            )

        array = np.array_split(array, WORLD_SIZE)[RANK]

    return array


def prepare(*objs, even: bool = False) -> list:
    """
    Get an specific partition for every object per process.

    Args:
        objs (`Any`):
            Objects to prepare.
        even (`bool`, *optional*, defaults to `False`):
            Wether to create even partitions across all processes.

    Returns:
        `list`: List of rank-specific object partitions.
    """
    prepared = []
    for obj in objs:
        if isinstance(obj, list):
            prepared.append(prepare_list(obj, even=even))
        elif isinstance(obj, pd.DataFrame):
            prepared.append(prepare_dataframe(obj, even=even))
        elif hasattr(obj, "__len__"):
            prepared.append(prepare_array(obj, even=even))
        else:
            prepared.append(obj)

    return prepared if len(prepared) > 1 else prepared[0]


def divide_into_batches(lst, batch_size):
    return [lst[i : i + batch_size] for i in range(0, len(lst), batch_size)]


def iterbatch(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]
