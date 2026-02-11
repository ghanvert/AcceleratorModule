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

import datetime
import inspect
import os
import re
import subprocess
import sys
import warnings
from collections import defaultdict
from contextlib import contextmanager
from typing import Callable, Union

import torch

from .globals import MASTER_PROCESS, RANK, WORLD_SIZE
from .maps import _units_map


def is_url(string: str) -> bool:
    """
    Check if a string is a URL.

    Args:
        string (`str`):
            String to check.

    Returns:
        `bool`: Whether the string is a URL.
    """
    if string in ["localhost", "127.0.0.1"]:
        return True

    url_regex = re.compile(
        r"^(?:http|ftp)s?://"
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"
        r"localhost|"
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"
        r"(?::\d+)?"
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    )
    return re.match(url_regex, string) is not None


def get_number_and_unit(string: str) -> tuple[int, Union[str, None]]:
    """
    Get the number and unit from a string.

    Args:
        string (`str`):
            String to parse.

    Returns:
        `tuple[int, str | None]`: Number and unit.
    """
    match = re.match(r"^(\d+)(\D+)?$", string.strip())

    if match:
        number = int(match.group(1))
        text = match.group(2).strip() if match.group(2) else ""
    else:
        number = 1
        text = string.strip()

    unit = None
    # Try to match the text (case-insensitive) to a value in _units_map
    for k, v in _units_map.items():
        if text.lower() in [u.lower() for u in v]:
            unit = k
            break
        # Also allow direct match to the key (e.g., "B" for "B")
        if text.lower() == k.lower():
            unit = k
            break

    # If text is not empty and unit is still None, return text as unit
    if text and unit is None:
        unit = text

    return number, unit


@contextmanager
def suppress_print_and_warnings(verbose=False):
    if not verbose:
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                yield
            finally:
                sys.stdout.close()
                sys.stdout = original_stdout
    else:
        yield


def combine_dicts(*dicts: list[dict]) -> dict:
    """
    Combine multiple dictionaries into a single dictionary.

    Args:
        *dicts (`list`):
            Dictionaries to combine.

    Returns:
        `dict`: Combined dictionary.
    """
    combined = {}
    for d in dicts:
        combined.update(d)
    return combined


def filter_kwargs(kwargs: dict, fn: Callable) -> dict:
    """
    Filter keyword arguments to only include those that are valid for a function.

    Args:
        kwargs (`dict`):
            Keyword arguments to filter.
        fn (`Callable`):
            Function to filter keyword arguments for.

    Returns:
        `dict`: Filtered keyword arguments.
    """
    try:
        return {k: v for k, v in kwargs.items() if k in fn.__init__.__code__.co_varnames}
    except AttributeError:
        signature = inspect.signature(fn)
        parameters = list(signature.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in parameters}


def print_gpu_users_by_device():
    try:
        # get GPU UUID ↔ index mapping
        uuid_map = {}
        uuid_output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], encoding="utf-8"
        )
        for line in uuid_output.strip().split("\n"):
            index, uuid = line.strip().split(", ")
            uuid_map[uuid] = int(index)

        # get GPU UUID + PID list
        usage_output = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], encoding="utf-8"
        )
        usage = defaultdict(set)

        for line in usage_output.strip().split("\n"):
            if not line.strip():
                continue
            gpu_uuid, pid = line.strip().split(", ")
            try:
                user = subprocess.check_output(["ps", "-o", "user=", "-p", pid], encoding="utf-8").strip()
                gpu_index = uuid_map.get(gpu_uuid, f"UUID:{gpu_uuid}")
                usage[gpu_index].add(user)
            except subprocess.CalledProcessError:
                continue

        rprint("Users using GPU(s):", start_char="")
        # print per-GPU usage
        num_gpus = max(uuid_map.values(), default=-1) + 1
        for gpu_idx in range(num_gpus):
            users = usage.get(gpu_idx, set())
            user_list = ", ".join(users) if users else "(idle)"
            rprint(f"GPU {gpu_idx}: {user_list}", start_char="")

    except subprocess.CalledProcessError as e:
        rprint("Error querying GPU usage:", e)


def get_time_prefix():
    return "[" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3] + "]"


def rprint(*args, rank: int = 0, add_time_prefix: bool = True, start_char: str = "\n", **kwargs):
    """Print on a specific rank (default is main process)."""
    if rank == RANK:
        if add_time_prefix:
            print(start_char, f"{get_time_prefix()} ", *args, **kwargs, sep="")
        else:
            print(start_char, *args, **kwargs, sep="")


def _breakpoint(rank: int = 0, *args, **kwargs):
    """
    Call `breakpoint(...)` on a specific rank (default is main process). Other ranks will wait.

    Args:
        rank (`int`):
            Rank to breakpoint on.
        *args:
            Arguments to pass to `breakpoint()`.
        **kwargs:
            Keyword arguments to pass to `breakpoint()`.
    """

    if rank == RANK:
        print(f"`breakpoint()` called on rank {rank}.")
        print("Type 'up' to go to the corresponding frame.")
        print("Type 'c' to continue execution.\n")
        breakpoint(*args, **kwargs)

    from .. import accelerator

    accelerator.wait_for_everyone()


def dist_breakpoint(*ranks: int):
    """
    Call `breakpoint()` on a list of ranks. If no ranks are provided, breakpoint on all ranks in ascending order.

    Args:
        *ranks (`list[int]`):
            Ranks to breakpoint on.
    """
    if len(ranks) == 0:
        ranks = [i for i in range(WORLD_SIZE)]

    from .. import accelerator

    if MASTER_PROCESS:
        print(f"Calling `breakpoint()` on ranks {ranks}.")
        print("Type 'up' to go to the corresponding frame.")
        print("Type 'c' to continue to the next rank or to continue execution.\n")

    for rank in ranks:
        if rank == RANK:
            print(f"`breakpoint()` called on rank {rank}.\n")
            breakpoint()
        accelerator.wait_for_everyone()


def global_randint(a: int = 0, b: int = 2147483647) -> int:
    """
    Return random integer in range [a, b], including both end points. This function ensures that all processes
    receive the same random number.

    Args:
        a (`int`, *optional*, defaults to 0):
            Lower limit.
        b (`int`, *optional*, defaults to 2147483647):
            Upper limit.
    """
    from .. import accelerator

    random_number = torch.randint(a, b, (1,), device=accelerator.device)
    return accelerator.gather(random_number)[0].item()
