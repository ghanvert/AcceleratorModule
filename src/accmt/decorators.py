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


def on_main_process(function):
    """Run on main process (RANK = 0)."""
    from . import accelerator

    return accelerator.on_main_process(function)


def on_last_process(function):
    """Run on last process (RANK = WORLD_SIZE - 1)."""
    from . import accelerator

    return accelerator.on_last_process(function)


def on_local_main_process(function):
    """Run on local main process (LOCAL_RANK = 0)."""
    from . import accelerator

    return accelerator.on_local_main_process(function)


def on_local_process(function, local_process_index: int):
    """Run on a specific local process (LOCAL_RANK = `local_process_index`)."""
    from . import accelerator

    return accelerator.on_local_process(function, local_process_index)


def on_process(function, process_index: int):
    """Run on a specific process (RANK = `process_index`)."""
    from . import accelerator

    return accelerator.on_process(function, process_index)
