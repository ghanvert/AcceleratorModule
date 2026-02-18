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

import time
from contextlib import contextmanager
from typing import Any, Optional

from .utils import MASTER_PROCESS


class DebugTimings:
    """Class to debug timings."""

    def __init__(self, debug: bool = True):
        """
        Debug timings using a buffer and a cache. The buffer will hold initial record and last record, while
        cache will hold all values of total time of execution.

        Args:
            debug (`bool`, *optional*, defaults to `True`):
                Flag to allow debugging.
        """
        self.buffer: dict[Any, float] = dict() if debug else None
        self.cache: dict[Any, float] = dict() if debug else None
        self.debug = debug

    def time(self, buffer: Optional[Any] = None) -> Optional[float]:
        """
        Record a time for a specific buffer. If time was already recorded, then it returns the total time of execution
        since last record. It also register that value into the cache.

        Args:
            buffer (`Any`, *optional*, defaults to `None`):
                A buffer to record values (start and end). Could be any type of key, like `str`.

        Returns:
            `float`: If recorded before, it returns the total time of execution. Otherwise,
            it returns `None`.
        """
        if not self.debug:
            return

        _time = time.time()
        if buffer not in self.buffer:
            self.buffer[buffer] = _time
        else:
            result = _time - self.buffer.pop(buffer)
            self.cache[buffer] = result
            return result

    @contextmanager
    def record_times(self, buffer: Any):
        """
        Context manager to record execution time.

        Args:
            buffer (`Any`):
                Buffer to record times.

        Example:
            ```
            with debug_timings.record_times("train_step"):
                # logic to record
            ```
        """
        if not self.debug:
            yield
            return

        self.time(buffer)
        try:
            yield
        finally:
            self.time(buffer)

    def reset_buffer(self):
        """Clear buffer."""
        if self.debug:
            self.buffer.clear()

    def reset_cache(self):
        """Clear cache."""
        if self.debug:
            self.cache.clear()

    def print_cache(self, *, reset_cache: bool = False):
        """
        Print cache nicely.

        Args:
            reset_cache (`bool`, *optional*, defaults to `False`):
                Wether to call `reset_cache` function after printing.
        """
        if not self.debug:
            return

        if MASTER_PROCESS:
            str_buffer = "\n"
            for i, (k, v) in enumerate(self.cache.items()):
                is_last_iter = i == len(self.cache) - 1
                str_buffer += f"{k}: {v:.3f} s."
                if not is_last_iter:
                    str_buffer += "   "

            print(str_buffer)

        if reset_cache:
            self.reset_cache()
