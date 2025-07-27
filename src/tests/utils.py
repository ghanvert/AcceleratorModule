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

import math

import torch

from accmt.utils import (
    clear_device_cache,
    combine_dicts,
    divide_into_batches,
    divide_list,
    drop_duplicates,
    filter_kwargs,
    get_array_partition,
    get_number_and_unit,
    get_seed,
    is_url,
    set_seed,
)


def test_combine_dicts():
    dict_a = {"a": 1}
    dict_b = {"b": 2}
    assert combine_dicts(dict_a, dict_b) == {"a": 1, "b": 2}


def test_clear_device_cache():
    clear_device_cache()
    clear_device_cache(garbage_collection=True)
    assert True


def test_divide_into_batches():
    n = 57
    batch_size = 6
    lst = [i for i in range(n)]
    divided_lst = divide_into_batches(lst, batch_size)
    assert len(divided_lst) == math.ceil(len(lst) / batch_size)


def test_divide_list():
    n = 57
    division = 6
    lst = [i for i in range(n)]
    divided_lst = divide_list(lst, division)
    assert len(divided_lst) == division


def test_drop_duplicates():
    n = 10
    tensor = torch.tensor([i for i in range(n)])
    mask = torch.tensor([i % 2 == 0 for i in range(n)])
    expected = torch.tensor([i for i in range(n) if i % 2 == 0])
    assert torch.all(drop_duplicates(tensor, mask) == expected).item()


def test_filter_kwargs():
    def fn(a, b, c):
        return a + b + c

    kwargs = {"a": 1, "b": 2, "c": 3, "d": 4}
    assert filter_kwargs(kwargs, fn) == {"a": 1, "b": 2, "c": 3}


def test_get_array_partition():
    n = 10
    array = torch.tensor([i for i in range(n)])
    assert torch.all(get_array_partition(array, 5, 3) == torch.tensor([6, 7])).item()


def test_is_url():
    assert is_url("https://www.google.com") and not is_url("not a url/")


def test_get_number_and_unit():
    assert get_number_and_unit("1000") == (1000, None)
    assert get_number_and_unit("1000B") == (1000, "B")


def test_seed():
    seed = 123
    assert get_seed() is None
    set_seed(seed)
    assert get_seed() == seed
