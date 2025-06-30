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
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from typing_extensions import Any, Literal, override

from .curriculum import _CurriculumLearning
from .states import TrainingState
from .tracker import BaseTracker


class AcceleratorModule(ABC):
    """
    Super class to define training and validation logic without the need
    to write a training loop.

    The constructor of this class must implement `self.model`, specifying the model
    from `torch.nn.Module`. `self.teacher` is also a reserved property for teacher-student
    approaches.
    """

    accelerator: Accelerator = None
    tracker: BaseTracker = None
    log_every: int = None
    state: TrainingState = None
    device: torch.device = None
    _implemented_collate_fn_train = False
    _implemented_collate_fn_val = False
    _extended = False
    model: nn.Module = None
    teacher: Optional[nn.Module] = None
    optimizer: Optimizer = None
    scheduler: LRScheduler = None
    _prepared: bool = False

    @override
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Defines the flow of data."""

    @override
    def training_step(self, batch: Any) -> torch.Tensor:
        """Defines the training logic. Must return a loss tensor (scalar)."""

    @override
    def validation_step(self, key: str, batch: Any) -> Union[dict, torch.Tensor]:
        """
        Defines the validation logic. Must return a dictionary containing
        each metric with corresponding arguments, and also the loss value in the dictionary.

        Example:
            ```
            # format is ==> "metric": (predictions, targets, ...)
            return {
                "loss": validation_loss_tensor, # (scalar tensor)
                # with additional metrics:
                "accuracy": (accuracy_predictions, accuracy_targets),
                "bleu": (bleu_predictions, bleu_targets)
            }
            ```
        """

    @override
    def test_step(self, batch: Any) -> Union[dict, torch.Tensor]:
        """
        Defines the test logic. Must return a dictionary containing
        each metric with corresponding arguments. This function is similar to `validation_step`,
        but it is used for testing using the `Evaluator` class.

        Example:
            ```
            # format is ==> "metric": (predictions, targets, ...)
            return {
                "accuracy": (accuracy_predictions, accuracy_targets),
                "...": (..., ...)
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
    def get_optimizer(self) -> Optimizer:
        """Defines a custom PyTorch optimizer logic here."""

    @override
    def get_scheduler(self, optimizer: Optimizer, steps_per_epoch: int, epochs: int) -> LRScheduler:
        """Defines a custom PyTorch scheduler logic here."""

    @override
    def get_train_dataloader(
        self, dataset: Union[Dataset, list[tuple[int, Dataset]], _CurriculumLearning]
    ) -> Union[DataLoader, list[tuple[int, DataLoader]]]:
        """
        Defines a custom PyTorch DataLoader class for training. In case of returning a `list` of tuples,
        the first element of each tuple represents the maximum step for each dataset, and the second element
        is the `DataLoader` for that dataset. For simple definitions of curriculum learning, you can use an instance of
        `StepsCurriculum`, `RangeCurriculum` or `RatioCurriculum` from `accmt.curriculum`.

        Must return a `DataLoader` or a `list` of tuples of `(max_step, DataLoader)`.
        """

    @override
    def get_validation_dataloader(
        self, dataset: Union[Dataset, dict[int, Dataset], list[Dataset]]
    ) -> Union[DataLoader, dict[int, DataLoader], list[DataLoader]]:
        """Defines a custom PyTorch DataLoader class for validation."""

    def log(
        self, values: dict[str, Union[torch.Tensor, float, int]], step: int, reduction: Literal["sum", "mean"] = "mean"
    ):
        """
        Log metrics to the tracker every N steps (defined in `Trainer`). If you want to apply any other logic,
        consider using `self.tracker.log` directly. This function will reduce tensors across all processes and only
        the main process will log the metrics.

        Args:
            values (`dict`):
                Dictionary of metrics to log. If values are tensors, they will be reduced across all processes. If
                values are not tensors, the ones from the main process will be logged.
            step (`int`):
                Step number to log the metrics. Can access `self.state.global_step` to log the current step,
                `self.state.train_step` or `self.state.val_step`.
            reduction (`str`, *optional*, defaults to `mean`):
                Reduction method to apply to tensors. Available options are `sum` and `mean`. Only applicable if
                values are tensors.
        """
        if step % self.log_every == 0:
            self.log_(values, step, reduction)

    def log_(
        self, values: dict[str, Union[torch.Tensor, float, int]], step: int, reduction: Literal["sum", "mean"] = "mean"
    ):
        """
        Log metrics to the tracker ignoring the `log_every` property. If you want to apply any other logic,
        consider using `self.tracker.log` directly. This function will reduce tensors across all processes and only
        the main process will log the metrics.

        Args:
            values (`dict`):
                Dictionary of metrics to log. If values are tensors, they will be reduced across all processes. If
                values are not tensors, the ones from the main process will be logged.
            step (`int`):
                Step number to log the metrics. Can access `self.state.global_step` to log the current step,
                `self.state.train_step` or `self.state.val_step`.
            reduction (`str`, *optional*, defaults to `mean`):
                Reduction method to apply to tensors. Available options are `sum` and `mean`. Only applicable if
                values are tensors.
        """
        values = {
            k: (self.accelerator.reduce(v.detach(), reduction=reduction).item() if isinstance(v, torch.Tensor) else v)
            for k, v in values.items()
        }
        self.tracker.log(values, step=step, run_id=self.tracker.run_id)

    def __init_subclass__(cls, **kwargs):
        # check collate functions
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

    def freeze(self, module: nn.Module):
        """
        Freeze all parameters inside a module.

        Args:
            module (`nn.Module`):
                Module where all parameters will have `requires_grad` set to `False`.
        """
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze(self, module: nn.Module):
        """
        Unfreeze all parameters inside a module.

        Args:
            module (`nn.Module`):
                Module where all parameters will have `requires_grad` set to `True`.
        """
        for param in module.parameters():
            param.requires_grad = True

    def pad(
        self,
        tensor: Union[torch.Tensor, list[torch.Tensor], tuple[torch.Tensor, ...]],
        value: Union[int, float],
        padding: Optional[Literal["max_length", "longest"]] = None,
        max_length: Optional[int] = None,
        side: Literal["left, right"] = "right",
        op: Optional[Union[str, Callable]] = None,
    ) -> Union[torch.Tensor, list[torch.Tensor], tuple[torch.Tensor, ...]]:
        """
        Pad last dimension of tensors to a given 'max_length' or to the longest tensor in an iterable (`tuple` or `list`).

        Args:
            tensor (`torch.Tensor`, `list` or `tuple`):
                Single tensor or an iterable of tensors to be padded.
            value (`int` or `float`):
                Constant value to be added when padding.
            padding (`str`, *optional*, defaults to `None`):
                Padding strategy to apply. `longest` means that all tensors in an iterable will be padded to
                the longest tensor, and `max_length` will pad all tensors to a given `max_length`. **NOTE**: A single
                tensor can only be padded to `max_length`. If padding is not specified, its value will default to
                `longest` for iterables and `max_length` for single tensors.
            max_length (`int`, *optional*, defaults to `None`):
                Max length for tensors to calculate remaining padding amount. This applies only when `padding` is set to
                `max_length` or `tensor` is a single tensor.
            side (`str`, *optional*, defaults to `right`):
                Padding side. Available options are `right` and `left`.
            op (`str`, *optional*, defaults to `None`):
                PyTorch operation to do after tensors are padded. Options can be `stack`, `cat` or a function. Only applicable
                for iterable of tensors.

        Returns:
            (`torch.Tensor`, `list` or `tuple`): Padded tensors.
        """
        _type = type(tensor)
        is_iterable = _type in {list, tuple}
        if _type is torch.Tensor or (is_iterable and len(tensor) == 1):
            if is_iterable:
                tensor = tensor[0]

            if tensor.ndim == 0:
                tensor.unsqueeze_(0)
            # if it's a single tensor, pad to 'max_length' and ignore 'padding'
            if max_length is None:
                self.accelerator.end_training()
                raise ValueError("When padding a single tensor, you must provide 'max_length'.")

            padding = max_length - tensor.size(-1)
            if padding < 0:
                raise RuntimeError("'pad' function is intended for padding and not truncation.")

            if side == "right":
                output = F.pad(tensor, pad=(0, padding), mode="constant", value=value)
            elif side == "left":
                output = F.pad(tensor, pad=(padding, 0), mode="constant", value=value)
            else:
                raise ValueError("'side' argument must be either 'left' or 'right'.")

            return _type(output) if is_iterable else output
        else:
            # if it's an iterable of tensors, pad to 'padding', and if 'padding' is not specified,
            # pad to 'longest'.
            padding = padding if padding is not None else "longest"
            if padding == "max_length":
                if max_length is None:
                    raise ValueError("Must provide 'max_length' argument when padding = 'max_length'.")

                _max_length = max_length
            else:
                _max_length = max(x.size(-1) for x in tensor)

            kwargs = {"value": value, "max_length": _max_length, "side": side}
            for x in tensor:
                x.data = self.pad(x, **kwargs)

            if op is not None:
                tensor = getattr(torch, op)(tensor) if isinstance(op, str) else op(tensor)

            return tensor  # objects inside iterable modified

    def compile(self):
        """
        Compile the model and teacher. At this stage, models are already on the correct device.
        """
        self.model = torch.compile(self.model)
        if self.teacher is not None:
            self.teacher = torch.compile(self.teacher)

    def before_eval(self):
        """
        This function is called before the evaluation loop.
        """
        pass

    def after_eval(self):
        """
        This function is called after the evaluation loop.
        """
        pass


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
        self.accelerator.backward(loss, **kwargs)

    def step_optimizer(self):
        self.optimizer.step()

    def step_scheduler(self):
        self.scheduler.step()

    def step(self):
        """Step optimizer and scheduler (in that order). If there is no scheduler, it will be ignored."""
        self.step_optimizer()
        if self.scheduler is not None:
            self.step_scheduler()

    def zero_grad(self, set_to_none: bool = True):
        """
        Call optimizer's 'zero_grad' operation to reset gradients.

        Args:
            `set_to_none` (`bool`, *optional*, defaults to `True`):
                Set gradients to `None` instead of `0`.
        """
        self.optimizer.zero_grad(set_to_none=set_to_none)

    @override
    def training_step(self, batch: Any):
        pass

    def __init_subclass__(cls, **kwargs):
        # No call to super(), so it suppresses the behavior.
        pass
