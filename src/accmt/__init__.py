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

import os
from datetime import timedelta

import torch
from accelerate import Accelerator, DataLoaderConfiguration, DistributedType, InitProcessGroupKwargs
from accelerate.utils import tqdm

from .callbacks import Callback
from .collate_fns import DataCollatorForLanguageModeling, DataCollatorForLongestSequence, DataCollatorForSeq2Seq
from .dataloader_samplers import TemperatureSampler
from .decorators import on_last_process, on_local_main_process, on_local_process, on_main_process, on_process
from .hyperparameters import HyperParameters, Optimizer, Scheduler
from .modules import AcceleratorModule, ExtendedAcceleratorModule
from .monitor import Monitor
from .tracker import Aim, ClearML, CometML, DVCLive, MLFlow, TensorBoard, WandB
from .trainer import Trainer
from .utility import IS_CPU, IS_GPU, prepare, prepare_array, prepare_dataframe
from .utils import _precision_map, get_seed, set_seed


def allow_tf32(flag=True):
    """Enable or disable the use of TensorFloat32."""
    torch.set_float32_matmul_precision("high" if flag else "highest")


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
    if accelerator.distributed_type != DistributedType.MULTI_CPU:
        """Function to auto cast all tensors to the corresponding precision (based on Mixed Precision)."""
        return tuple(tensor.to(precision) for tensor in tensors) if len(tensors) > 1 else tensors[0].to(precision)

    return tensors


def autocast_(*tensors: torch.Tensor):
    """Inplace function to auto cast all tensors to the corresponding precision (based on Mixed Precision)."""
    if accelerator.distributed_type != DistributedType.MULTI_CPU:
        for tensor in tensors:
            tensor.data = tensor.to(precision)
