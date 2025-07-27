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
from typing import Callable

from .globals import RANK
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


def get_number_and_unit(string: str) -> tuple[int, str]:
    """
    Get the number and unit from a string.

    Args:
        string (`str`):
            String to parse.

    Returns:
        `tuple[int, str]`: Number and unit.
    """
    match = re.match(r"(\d+)(\D+)", string)

    if match:
        number = int(match.group(1))
        text = match.group(2).strip().lower()
    else:
        number = 1
        text = string.strip().lower()

    unit = None
    for k, v in _units_map.items():
        if text in v:
            unit = k
            break

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
        breakpoint(*args, **kwargs)

    from .. import accelerator

    accelerator.wait_for_everyone()
