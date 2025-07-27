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

from typing import Optional

from accelerate.utils import set_seed as accelerate_set_seed


SEED: Optional[int] = None


def set_seed(seed: int):
    """Set a global seed for `random`, `numpy` and `torch`."""
    accelerate_set_seed(seed)
    global SEED
    SEED = seed


def get_seed(default: Optional[int] = None) -> Optional[int]:
    """
    Get global seed. If it was not set, this will return `default`.

    Returns:
        `int` or `None`: Global seed.
    """
    global SEED
    return SEED if SEED is not None else default
