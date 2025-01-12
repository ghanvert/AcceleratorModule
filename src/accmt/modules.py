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

from abc import ABC
from typing import Optional, Union

import torch
from accelerate import Accelerator
from typing_extensions import Any, override

from .states import TrainingState


class AcceleratorModule(ABC):
    """
    Super class to define training and validation logic without the need
    to write a training loop.

    The constructor of this class must implement `self.model`, specifying the model
    from `torch.nn.Module`. `self.teacher` is also a reserved property for teacher-student
    approaches.

    Methods:
        `forward` (*optional*):
            Defines the flow of data of model. If not implemented, `__call__`
            will not be possible (e.g. `self(...)`). Should return the model output.
        `training_step` (*optional*):
            Defines the training logic. Must return a loss `torch.Tensor` (scalar).
        `validation_step` (*optional*):
            Defines the validation logic. Must return a loss `torch.Tensor` (scalar).
            If not implemented, no validation will be executed.
        `collate_fn_train` (*optional*):
            Defines the collator function for train DataLoader.
        `collate_fn_val` (*optional*):
            Defines the collator function for validation DataLoader.
        `get_optimizer` (*optional*):
            Defines the optimizer. Must return the optimizer itself.
        `get_scheduler` (*optional*):
            Defines the scheduler. Must return the scheduler itself.
        `get_train_dataloader` (*optional*):
            Defines the train DataLoader. Must return a torch `DataLoader`.
        `get_validation_dataloader` (*optional*):
            Defines the validation DataLoader. Must return a torch `DataLoader`.

    Special methods (no implementation required):
        `__call__`:
            When calling this module, it will execute `forward` method.
        `__repr__`:
            When reproducing this module (e.g. Jupyter Notebook cell), this will print
            the model structure from `torch.nn.Module` specified in `self.model`.
        `__str__`:
            When printing this module or using it as a `str` type, this will represent
            the `torch.nn.Module` specified in `self.model`.
        `__len__`:
            When casting this module with `len` Python function, it will return the
            number of parameters of the base model specified in `self.model` from
            `torch.nn.Module`.
    """

    _implemented_collate_fn_train = False
    _implemented_collate_fn_val = False
    _accelerator: Accelerator = None
    _log_every: int = 1
    _extended = False
    state: TrainingState = None
    device: torch.device = None
    status_dict: dict = None
    batch_size: Union[int, tuple[int, int]] = None

    @override
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Defines the flow of data."""

    @override
    def training_step(self, batch: Any) -> torch.Tensor:
        """Defines the training logic. Must return a loss tensor (scalar)."""

    @override
    def validation_step(self, batch: Any) -> dict:
        """
        Defines the validation logic. Must return a dictionary containing
        each metric with predictions and targets, and also the loss value in the dictionary.

        Example:
            ```
            # format is ==> "metric": (predictions, targets)
            return {
                "loss": validation_loss_tensor, # (scalar tensor)
                # with additional metrics:
                "accuracy": (accuracy_predictions, accuracy_targets),
                "bleu": (bleu_predictions, bleu_targets)
            }
            ```
        """

    @override
    def collate_fn_train(self, batch: list) -> Any:
        """Defines a collate function for PyTorch train DataLoader."""

    @override
    def collate_fn_val(self, batch: list) -> Any:
        """Defines a collate function for PyTorch validation DataLoader."""

    @override
    def get_optimizer(self, *args: Any, **kwargs: Any) -> Any:
        """Defines a custom PyTorch optimizer logic here."""

    @override
    def get_scheduler(self, optimizer: Any, steps_per_epoch: int, epochs: int) -> Any:
        """Defines a custom PyTorch scheduler logic here."""

    @override
    def get_train_dataloader(self, *args: Any, **kwargs: Any) -> Any:
        """Defines a custom PyTorch DataLoader class for training."""

    @override
    def get_validation_dataloader(self, *args: Any, **kwargs: Any) -> Any:
        """Defines a custom PyTorch DataLoader class for validation."""

    def log(self, values: dict, log_kwargs: dict | None = {}):
        if self._accelerator.is_main_process:
            train_or_eval = "global_step" if self.model.training else "eval_global_step"
            if (self.status_dict[train_or_eval] + 1) % self._log_every == 0:
                self._accelerator.log(values, step=self.status_dict[train_or_eval], log_kwargs=log_kwargs)

    def __init_subclass__(cls, **kwargs):
        if (
            cls.training_step == AcceleratorModule.training_step
            and cls.validation_step == AcceleratorModule.validation_step
        ):
            raise TypeError(
                "Subclasses of 'Trainer' must override 'training_step' and/or " "'validation_step' methods."
            )

        if cls.collate_fn_train != AcceleratorModule.collate_fn_train:
            cls._implemented_collate_fn_train = True

        if cls.collate_fn_val != AcceleratorModule.collate_fn_val:
            cls._implemented_collate_fn_val = True

        super().__init_subclass__(**kwargs)

    def __call__(self, *args: Any, **kwargs: Any):
        return self.forward(*args, **kwargs)

    def __repr__(self):
        return self.model

    def __str__(self):
        return self.model.__repr__()

    def __len__(self):
        return sum(p.numel() for p in self.model.parameters())

    @classmethod
    def from_hf(cls, path: str, type: Union[str, Any] = None, **kwargs: Optional[Any]):
        """
        Build a custom AcceleratorModule for HuggingFace's transformers library. It simply replaces the following standard:

        ```
        class Module(AcceleratorModule):
            def __init__(self):
                self.model = AutoModel.from_pretrained(path, **kwargs)

            def training_step(self, batch):
                return self.model(**batch).loss

            def validation_step(self, batch):
                return {"loss": self.model(**batch).loss}
        ```

        Args:
            path (`str`):
                Path for HuggingFace model.
            type (`str` or `Any`):
                Model type in transformers library. It can be the class itself or a string (no need for imports).
            kwargs (`Any`):
                Keyword arguments for `from_pretrained` function for model initialization.
        """
        if isinstance(type, str):
            import importlib

            module = importlib.import_module("transformers")
            type = getattr(module, type)
        elif type is None:
            from transformers import AutoModel

            type = AutoModel

        class Module(AcceleratorModule):
            def __init__(self):
                self.model = type.from_pretrained(path, **kwargs)

            def training_step(self, batch):
                return self.model(**batch).loss

            def validation_step(self, batch):
                return self.model(**batch).loss

        return Module()


class ExtendedAcceleratorModule(AcceleratorModule):
    """
    Extended module from `AcceleratorModule` to enhance `training_step` function. This
    means that the backpropagation part must be done manually.

    Example:
        ```
        class Module(ExtendedAcceleratorModule):
            # other logic remains the same

            def training_step(self, batch):
                loss = ...
                self.backward(loss)
                self.step_optimizer()
                self.step_scheduler()

                return loss  # loss will only be used to log metrics.
        ```

    NOTE: `grad_accumulation_steps` in `fit` function from `Trainer` will not work. If you want to accumulate gradients
    and then backpropagate, you may want to make use of `self.status_dict["epoch_step"]`.
    """

    _extended = True

    def backward(self, loss: torch.Tensor, **kwargs):
        """
        Performs backward operation.

        Args:
            `loss` (`torch.Tensor`):
                Scalar loss tensor to backward.
            `kwargs` (`Any`):
                Extra arguments to be passed to 'accelerator.backward' function.
        """
        self._accelerator.backward(loss, **kwargs)

    def step_optimizer(self):
        self.state.optimizer.step()

    def step_scheduler(self):
        self.state.scheduler.step()

    def step(self):
        """Step optimizer and scheduler (in that order). If there is no scheduler, it will be ignored."""
        self.step_optimizer()
        if self.state.scheduler is not None:
            self.step_scheduler()

    def zero_grad(self, set_to_none: bool = True):
        """
        Call optimizer's 'zero_grad' operation to reset gradients.

        Args:
            `set_to_none` (`bool`, *optional*, defaults to `True`):
                Set gradients to `None` instead of `0`.
        """
        self.state.optimizer.zero_grad(set_to_none=set_to_none)

    @override
    def training_step(self, batch: Any):
        pass
