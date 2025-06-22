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

from datetime import timedelta

import torch
from accelerate import Accelerator, DataLoaderConfiguration, DistributedType, InitProcessGroupKwargs

from .callbacks import Callback
from .utils import _precision_map, get_seed, is_transformers_available, set_seed


if is_transformers_available():
    from .collate_fns import (
        DataCollatorForLanguageModeling,
        DataCollatorForPermutationLanguageModeling,
        DataCollatorForSeq2Seq,
        DataCollatorForTokenClassification,
        DataCollatorForWholeWordMask,
        DataCollatorWithPadding,
    )

from .dataloader_samplers import TemperatureSampler
from .decorators import on_last_process, on_local_main_process, on_local_process, on_main_process, on_process
from .evaluator import Evaluator
from .hp_search import HyperParameterSearch
from .hyperparameters import HyperParameters, Optimizer, Scheduler
from .modules import AcceleratorModule, ExtendedAcceleratorModule
from .monitor import Monitor
from .tqdm import tqdm
from .trainer import Trainer, __version__
from .utility import IS_CPU, IS_GPU, prepare, prepare_array, prepare_dataframe


def is_tf32_supported() -> bool:
    """
    Check if TensorFloat32 is supported. Implementation is identical to `torch.cuda.is_tf32_supported`
    in `torch` library (v2.6.0+).
    """
    # Check for ROCm.  If true, return false, since PyTorch does not currently support
    # tf32 on ROCm.
    if torch.version.hip:
        return False

    # Otherwise, tf32 is supported on CUDA platforms that natively (i.e. no emulation)
    # support bfloat16.
    return torch.cuda.is_bf16_supported(including_emulation=False)


def allow_tf32(flag=True):
    """Enable or disable the use of TensorFloat32 (if supported)."""
    if is_tf32_supported():
        torch.set_float32_matmul_precision("high" if flag else "highest")


if IS_GPU and torch.cuda.is_available():
    allow_tf32()

_init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=86400))
_dataloader_config = DataLoaderConfiguration(use_seedable_sampler=True)
accelerator = Accelerator(
    kwargs_handlers=[_init_kwargs],
    dataloader_config=_dataloader_config,
    step_scheduler_with_optimizer=False,
    cpu=IS_CPU,
)

precision = _precision_map.get(accelerator.mixed_precision, torch.float32)


def autocast(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Function to auto cast all tensors to the corresponding precision (based on Mixed Precision)."""
    if accelerator.distributed_type != DistributedType.MULTI_CPU:
        return tuple(tensor.to(precision) for tensor in tensors) if len(tensors) > 1 else tensors[0].to(precision)

    return tensors


def autocast_(*tensors: torch.Tensor):
    """Inplace function to auto cast all tensors to the corresponding precision (based on Mixed Precision)."""
    if accelerator.distributed_type != DistributedType.MULTI_CPU:
        for tensor in tensors:
            tensor.data = tensor.to(precision)
