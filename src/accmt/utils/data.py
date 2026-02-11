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

from typing import Any

import numpy as np
import torch

from .availability import is_pandas_available
from .globals import LAST_PROCESS, RANK, WORLD_SIZE


def divide_list(lst: list, parts: int) -> list[list]:
    """
    Divide a list into a specified number of parts.

    Args:
        lst (`list`):
            List to divide.
        parts (`int`):
            Number of parts to divide the list into.

    Returns:
        `list`: List of parts.
    """
    k, m = divmod(len(lst), parts)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(parts)]


def get_array_partition(array: Any, n_partitions: int, get: int) -> Any:
    """
    Get an specific partition of an array-like object. This returns a view of the
    original object.

    Args:
        array (`Any`):
            Array-like.
        n_partitions (`int`):
            Number of partitions to create from the original array-like.
        get (`int`):
            Get the specific partition.

    Returns:
        `Any`: Specific partition (`get`) of the given array-like.
    """
    length = len(array)
    chunk_size, remainder = divmod(length, n_partitions)

    start = get * chunk_size + min(get, remainder)
    end = start + chunk_size + (1 if get < remainder else 0)

    return array[start:end]


def prepare_dataframe(df: Any, even: bool = False) -> tuple[Any, int]:
    """
    Get an specific partition of a Pandas DataFrame per process.

    Args:
        df (`pandas.DataFrame`):
            Pandas DataFrame.
        even (`bool`, *optional*, defaults to `False`):
            Wether to create even partitions across all processes.

    Returns:
        `list`: Rank-specific Pandas DataFrame partition.
    """
    if not is_pandas_available():
        raise ImportError("Pandas is not installed. Please install it with `pip install pandas`.")

    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise ValueError("`df` must be an instance of a Pandas DataFrame.")

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


def prepare_array(array: Any, even: bool = False) -> Any:
    """
    Get an specific partition of an array-like per process.

    Args:
        array (`Any`):
            Array-like.
        even (`bool`, *optional*, defaults to `False`):
            Wether to create even partitions across all processes.

    Returns:
        `Any`: Rank-specific array-like partition.
    """
    if not hasattr(array, "__len__"):
        raise ValueError("Values passed to 'prepare_array' must be array-like only (not scalar values).")

    if WORLD_SIZE > 1:
        array_type = type(array)
        if even:
            size, remainder = divmod(len(array), WORLD_SIZE)
            max_size = size + remainder
            diff = len(array) - max_size

            last_elem = array[-1:]
            padding_array = np.repeat(last_elem, diff, axis=0)
            array = np.concatenate([array, padding_array], axis=0)

        array = get_array_partition(array, n_partitions=WORLD_SIZE, get=RANK)
        if isinstance(array, np.ndarray):
            if array_type is torch.Tensor:
                array = torch.from_numpy(array)
            elif array_type is list:
                array = array.tolist()

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
        _processed = False
        if is_pandas_available():
            import pandas as pd

            if isinstance(obj, pd.DataFrame):
                prepared.append(prepare_dataframe(obj, even=even))
                _processed = True

        if not _processed:
            if hasattr(obj, "__len__"):
                prepared.append(prepare_array(obj, even=even))
            else:
                prepared.append(obj)

    return prepared if len(prepared) > 1 else prepared[0]


def divide_into_batches(lst: list, batch_size: int) -> list[list]:
    """
    Divide a list into batches of a specified size.

    Args:
        lst (`list`):
            List to divide.
        batch_size (`int`):
            Size of each batch.

    Returns:
        `list`: List of batches.
    """
    return [lst[i : i + batch_size] for i in range(0, len(lst), batch_size)]


def iterbatch(lst: list, batch_size: int) -> list[list]:
    """
    Iterate over a list in batches of a specified size.

    Args:
        lst (`list`):
            List to iterate over.
        batch_size (`int`):
            Size of each batch.

    Returns:
        `list`: List of batches.
    """
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]
