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

from .callbacks import Callback
from .utils import IS_CPU, IS_GPU, __version__, _precision_map, is_tf32_supported, is_transformers_available


if is_transformers_available():
    from .collate_fns import (
        DataCollatorForLanguageModeling,
        DataCollatorForPermutationLanguageModeling,
        DataCollatorForSeq2Seq,
        DataCollatorForTokenClassification,
        DataCollatorForWholeWordMask,
        DataCollatorWithPadding,
    )

from .decorators import on_last_process, on_local_main_process, on_local_process, on_main_process, on_process
from .evaluator import Evaluator
from .hp_search import HyperParameterSearch
from .hyperparameters import HyperParameters, Optimizer, Scheduler
from .modules import AcceleratorModule, ExtendedAcceleratorModule
from .monitor import Monitor
from .tqdm import tqdm
from .trainer import Trainer


def allow_tf32(flag=True):
    """Enable or disable the use of TensorFloat32 (if supported)."""
    if is_tf32_supported():
        torch.set_float32_matmul_precision("high" if flag else "highest")


def allow_cudnn_sdpa(flag=True):
    """
    Enable or disable the cuDNN backend of `scaled_dot_product_attention`. cuDNN builds an execution plan per input
    shape, so with variable sequence lengths (e.g. dynamic padding) under bf16/fp16 most steps pay for a new plan.
    """
    torch.backends.cuda.enable_cudnn_sdp(flag)


if IS_GPU and torch.cuda.is_available():
    allow_tf32()
    allow_cudnn_sdpa(bool(int(os.environ.get("ACCMT_CUDNN_SDPA", "0"))))

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
