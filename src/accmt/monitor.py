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
from dataclasses import dataclass
from typing import Any, Optional, Union

import psutil
import torch
from accelerate import Accelerator

from .states import TrainingState
from .tracker import BaseTracker
from .utility import DEBUG_MODE, MASTER_PROCESS


@dataclass
class Monitor:
    """
    Class to set metrics to monitor during training using a tracker (if implemented).

    Args:
        learning_rate (`bool`, *optional*, defaults to `False`):
            Monitor learning rate.
        epoch (`bool`, *optional*, defaults to `True`):
            Monitor current epoch.
        train_loss (`bool`, *optional*, defaults to `True`):
            Monitor training loss.
        validation_loss (`bool`, *optional*, defaults to `True`):
            Monitor validation loss.
        accuracy (`bool`, *optional*, defaults to `True`):
            Monitor accuracy if implemented.
        grad_norm (`bool`, *optional*, defaults to `False`):
            This will enable monitoring for gradient normalization. This feature is not yet supported
            when running with DeepSpeed.
        gpu_utilization (`bool`, *optional*, defaults to `False`):
            Monitor GPU utilization in GB. It only reports GPU from main process (for now).
        cpu_utilization (`bool`, *optional*, defaults to `False`):
            Monitor CPU utilization in GB. It only reports CPU from main process (for now)
    """

    def __init__(
        self,
        learning_rate: bool = False,
        epoch: bool = True,
        train_loss: bool = True,
        validation_loss: bool = True,
        additional_metrics: bool = True,
        grad_norm: bool = False,
        gpu_utilization: bool = False,
        cpu_utilization: bool = False,
    ):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.train_loss = train_loss
        self.validation_loss = validation_loss
        self.additional_metrics = additional_metrics
        self.grad_norm = grad_norm
        self.gpu_utilization = gpu_utilization
        self.cpu_utilization = cpu_utilization
        self.accelerator: Accelerator = None
        self.tracker: BaseTracker = None
        self.train_loss_name: str = None
        self.validation_loss_name: str = None
        self.state: TrainingState = None
        self._tracking = MASTER_PROCESS and DEBUG_MODE < 1

    @classmethod
    def from_config(cls, config: Union[str, dict]):
        """
        Load a monitor configuration from a file or a dictionary.

        Args:
            config (`str` or `dict`):
                Path to a file or dictionary containing kwargs for Monitor constructor. The file can
                be YAML or JSON.
        """
        assert config is None or isinstance(config, (str, dict)), f"{config} is not of type 'str' or 'dict'."
        if isinstance(config, str):
            import yaml

            config = yaml.safe_load(open(config))
        elif config is None:
            config = {}

        return Monitor(**config)

    def _set_extra(
        self,
        accelerator: Accelerator,
        state: TrainingState,
        train_loss_name: str,
        validation_loss_name: str,
        tracker: BaseTracker,
    ):
        self.accelerator = accelerator
        self.state = state
        self.train_loss_name = train_loss_name
        self.validation_loss_name = validation_loss_name
        self.tracker = tracker

    def log(self, name: str, value: Union[int, float, torch.Tensor], run_id: Optional[str] = None):
        if isinstance(value, torch.Tensor):
            value = value.item()

        self.tracker.log({name: value}, step=self.state.global_step, run_id=run_id)

    def log_values(self, values: dict[str, Any], run_id: Optional[str] = None):
        values = {k: (v if not isinstance(v, torch.Tensor) else v.item()) for k, v in values.items()}
        self.tracker.log(values, step=self.state.global_step, run_id=run_id)

    def log_learning_rate(self, value: Union[int, float, torch.Tensor], run_id: Optional[str] = None):
        if self._tracking and self.learning_rate:
            self.log("learning_rate", value, run_id=run_id)

    def log_epoch(self, value: Union[int, float, torch.Tensor], run_id: Optional[str] = None):
        if self._tracking and self.epoch:
            self.log("epoch", value, run_id=run_id)

    def log_train_loss(self, value: Union[int, float, torch.Tensor], run_id: Optional[str] = None):
        if self._tracking and self.train_loss:
            self.log(self.train_loss_name, value, run_id=run_id)

    def log_validation_loss(self, value: Union[int, float, torch.Tensor], run_id: Optional[str] = None):
        if self._tracking and self.validation_loss:
            self.log(self.validation_loss_name, value, run_id=run_id)

    def log_additional_metrics(self, values: dict[str, Any], run_id: Optional[str] = None):
        if self._tracking and self.additional_metrics:
            self.log_values(values, run_id=run_id)

    def log_gpu_utilization(self, run_id: Optional[str] = None):
        if self._tracking and self.gpu_utilization:
            device = self.accelerator.device
            memory_allocated = torch.cuda.memory_allocated(device)
            memory_reserved = torch.cuda.memory_reserved(device)
            total_memory = (memory_allocated + memory_reserved) / (1024**3)

            self.log("GPU_0", total_memory, run_id=run_id)

    def log_cpu_utilization(self, run_id: Optional[str] = None):
        if self._tracking and self.cpu_utilization:
            process = psutil.Process(os.getpid())
            cpu_mem = process.memory_info().rss / (1024**3)
            self.log("CPU_PROCESS_0", cpu_mem, run_id=run_id)

    def log_grad_norm(self, value: Union[int, float, torch.Tensor], run_id: Optional[str] = None):
        if self._tracking and self.grad_norm:
            self.log("grad_norm", value, run_id=run_id)

    def log_train_loss_and_grad_norm(
        self, train_loss: float, grad_norm: Optional[Union[torch.Tensor, float]] = None, run_id: Optional[str] = None
    ):
        """Fused functions to only report once to server."""
        _dict = {}
        if self._tracking and self.train_loss:
            _dict[self.train_loss_name] = train_loss

        if self._tracking and grad_norm is not None and self.grad_norm:
            _dict["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

        if self._tracking:
            self.tracker.log(_dict, step=self.state.global_step, run_id=run_id)
