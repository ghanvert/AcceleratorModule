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

from accelerate.utils.memory import clear_device_cache

from .availability import (
    _is_module_available,
    is_deepspeed_available,
    is_pandas_available,
    is_tf32_supported,
    is_transformers_available,
)
from .data import (
    divide_into_batches,
    divide_list,
    get_array_partition,
    iterbatch,
    prepare,
    prepare_array,
    prepare_dataframe,
)
from .distributed import (
    all_gather_dictionary,
    gather,
    gather_and_drop_duplicates,
    gather_into_single_process,
    gather_object,
)
from .globals import (
    ASYNC,
    ASYNC_HASH,
    ASYNC_TRAIN_GROUP,
    DEBUG_MODE,
    DIST_HASH,
    IS_CPU,
    IS_GPU,
    LAST_PROCESS,
    MASTER_PROCESS,
    RANK,
    WORLD_SIZE,
    __version__,
)
from .maps import _operator_map, _pandas_reader_map, _precision_map, _units_map
from .misc import (
    _breakpoint,
    combine_dicts,
    dist_breakpoint,
    filter_kwargs,
    get_number_and_unit,
    get_time_prefix,
    global_randint,
    is_url,
    print_gpu_users_by_device,
    rprint,
)
from .seed import SEED, get_seed, set_seed
from .tensor import drop_duplicates, pad
