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

from abc import ABC

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing_extensions import Any, override

from .modules import AcceleratorModule
from .states import TrainingState


class Callback(ABC):
    """
    Callback module containing different callback functions for different
    stages of the traininig process.

    NOTE: Every callback function will run on every process. If you want your
    callback functions to only run on a single process, make sure to import
    `accmt.decorators` for different function decorators.

    Attributes:
        module (`AcceleratorModule`):
            Training module.
        trainer (`Trainer`):
            Defined `Trainer` class.
        state (`TrainingState`):
            Module's `TrainingState` class.

    Methods:
        on_fit_start (*optional*):
            Callback when training process starts.
        on_fit_end (*optional*):
            Callback when training process ends.
        on_before_backward (*optional*):
            Callback before engine's backward.
        on_after_backward (*optional*):
            Callback after engine's backward.
        on_before_optimizer_step (*optional*):
            Callback before optimizers steps.
        on_after_optimizer_step (*optional*):
            Callback after optimizer steps.
        on_before_scheduler_step (*optional*):
            Callback before scheduler steps:
        on_after_scheduler_step (*optional*):
            Callback after scheduler steps.
        on_before_zero_grad (*optional*):
            Callback before optimizer resets gradients.
        on_after_zero_grad (*optional*):
            Callback after optimizer resets gradients.
        on_runtime_error (*optional*):
            Callback when process raises a `RunTimeError` exception.
        on_cuda_out_of_memory (*optional*):
            Callback when process raises a `RunTimeError` exception with
            CUDA Out Of Memory.
        on_keyboard_interrupt (*optional*):
            Callback when process raises a `KeyboardInterrupt` exception.
        on_exception (*optional*):
            Callback when process raises any other `Exception` different than
            `RuntimeError` and `KeyboardInterrupt`
        on_resume (*optional*):
            Callback when resuming training process.
        on_save_checkpoint (*optional*):
            Callback when saving checkpoint.
        on_before_training_step (*optional*):
            Callback before `training_step` function.
        on_after_training_step (*optional*):
            Callback after `training_step` function.
        on_before_validation_step (*optional*):
            Callback before `validation_step` function.
        on_after_validation_step (*optional*):
            Callback after `validation_step` function.
        on_epoch_start (*optional*):
            Callback when an epoch starts.
        on_epoch_end (*optional*):
            Callback when an epoch ends.
        on_evaluation_start (*optional*):
            Callback when evaluation starts.
        on_evaluation_end (*optional*):
            Callback when evaluation ends.

    """

    module: AcceleratorModule = None
    trainer = None
    state: TrainingState = None

    @override
    def on_fit_start(self):
        """Callback when training process starts."""

    @override
    def on_fit_end(self):
        """Callback when training process ends."""

    @override
    def on_before_backward(self, loss: torch.Tensor):
        """
        Callback before engine's backward.

        Args:
            loss (`torch.Tensor`):
                Scalar loss tensor.
        """

    @override
    def on_after_backward(self):
        """Callback after engine's backward."""

    @override
    def on_before_optimizer_step(self, optimizer: Optimizer):
        """
        Callback before optimizers steps.

        Args:
            optimizer (`Optimizer`):
                Wrapped optimizer.
        """

    @override
    def on_after_optimizer_step(self, optimizer: Optimizer):
        """
        Callback after optimizer steps.

        Args:
            optimizer (`Optimizer`):
                Wrapped optimizer.
        """

    @override
    def on_before_scheduler_step(self, scheduler: LRScheduler):
        """
        Callback before scheduler steps:

        Args:
            scheduler (`LRScheduler`):
                Wrapped scheduler.
        """

    @override
    def on_after_scheduler_step(self, scheduler: LRScheduler):
        """
        Callback after scheduler steps.

        Args:
            scheduler (`LRScheduler`):
                Wrapped scheduler.
        """

    @override
    def on_before_zero_grad(self, optimizer: Optimizer):
        """
        Callback before optimizer resets gradients.

        Args:
            optimizer (`Optimizer`):
                Wrapped optimizer.
        """

    @override
    def on_after_zero_grad(self, optimizer: Optimizer):
        """
        Callback after optimizer resets gradients.

        Args:
            optimizer (`Optimizer`):
                Wrapped optimizer.
        """

    @override
    def on_runtime_error(self, exception: Exception):
        """
        Callback when process raises a `RunTimeError` exception.

        Args:
            exception (`Exception`):
                Raised exception.
        """

    @override
    def on_cuda_out_of_memory(self, exception: Exception):
        """
        Callback when process raises a `RunTimeError` exception with
        CUDA Out Of Memory.

        Args:
            exception (`Exception`):
                Raised exception.
        """

    @override
    def on_keyboard_interrupt(self, exception: Exception):
        """
        Callback when process raises a `KeyboardInterrupt` exception.

        Args:
            exception (`Exception`):
                Raised exception.
        """
        pass

    @override
    def on_exception(self, exception: Exception):
        """
        Callback when process raises any other `Exception` different than
        `RuntimeError` and `KeyboardInterrupt`

        Args:
            exception (`Exception`):
                Raised exception.
        """

    @override
    def on_resume(self):
        """Callback when resuming training process."""

    @override
    def on_save_checkpoint(self):
        """Callback when saving checkpoint."""

    @override
    def on_before_training_step(self, batch: Any):
        """
        Callback before `training_step` function.

        Args:
            batch (`Any`):
                Dataloader's batch.
        """

    @override
    def on_after_training_step(self):
        """Callback after `training_step` function."""

    @override
    def on_before_validation_step(self, batch: Any):
        """
        Callback before `validation_step` function.

        Args:
            batch (`Any`):
                Dataloader's batch.
        """

    @override
    def on_after_validation_step(self):
        """Callback after `validation_step` function."""

    @override
    def on_epoch_start(self):
        """Callback when an epoch starts."""

    @override
    def on_epoch_end(self):
        """Callback when an epoch ends."""

    @override
    def on_evaluation_start(self):
        """Callback when evaluation starts."""

    @override
    def on_evaluation_end(self):
        """Callback when evaluation ends."""
