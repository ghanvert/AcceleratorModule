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
import gc
import inspect
import operator
import os
import re
import sys
import warnings
from contextlib import contextmanager
from typing import Optional

import pandas as pd
import torch
from accelerate.utils import set_seed as accelerate_set_seed


units = {
    "epoch": {"epoch", "ep", "epochs", "eps"},
    "step": {"step", "st", "steps", "sts"},
    "eval": {"evaluation", "eval", "evaluations", "evals"},
}

_precision_map = {
    "no": torch.float32,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}

SEED: Optional[int] = None


def set_seed(seed: int):
    """Set a global seed for `random`, `numpy` and `torch`."""
    accelerate_set_seed(seed)
    global SEED
    SEED = seed


def get_seed(default: Optional[int] = None) -> Optional[int]:
    """Get global seed. If it was not set, this will return `default`."""
    global SEED
    return SEED if SEED is not None else default


def is_url(string):
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


def get_number_and_unit(string: str):
    match = re.match(r"(\d+)(\D+)", string)

    if match:
        number = int(match.group(1))
        text = match.group(2).strip().lower()
    else:
        number = 1
        text = string.strip().lower()

    unit = None
    for k, v in units.items():
        if text in v:
            unit = k
            break

    return number, unit


def combine_dicts(*dicts):
    combined = {}
    for d in dicts:
        combined.update(d)
    return combined


def divide_list(lst: list, parts: int):
    k, m = divmod(len(lst), parts)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(parts)]


def time_prefix():
    return "[" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3] + "]"


PANDAS_READER_MAP = {
    "csv": pd.read_csv,
    "xlsx": pd.read_excel,
    "xml": pd.read_xml,
    "feather": pd.read_feather,
    "parquet": pd.read_parquet,
    "pickle": pd.read_pickle,
    "pkl": pd.read_pickle,
}


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


def filter_kwargs(kwargs: dict, fn):
    try:
        return {k: v for k, v in kwargs.items() if k in fn.__init__.__code__.co_varnames}
    except AttributeError:
        signature = inspect.signature(fn)
        parameters = list(signature.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in parameters}


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


operator_map = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge, "==": operator.eq}
