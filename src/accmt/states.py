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

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import torch
from accelerate import Accelerator

from .utility import MASTER_PROCESS


@dataclass
class TrainingState:
    """
    General training state.

    Args:
        batch_iteration (`int`):
            Batch iteration index. This is incremented every time a train step is done. This ignores gradient
            accumulation. If gradient accumulation is not used, this is the same as `global_step`.
        global_step (`int`):
            Global step index. This is incremented every time a train step and a gradient accumulation step is done.
        train_step (`int`):
            Training step index inside a training loop (can be considered as batch index).
        val_step (`int`):
            Validation step index inside an evaluation loop (can be considered as batch index).
        epoch (`int`):
            Epoch index.
        is_end_of_epoch (`bool`):
            Flag to check if current state is at the end of an epoch.
        is_last_training_batch (`bool`):
            Flag to check if current state is processing the last training batch.
        is_last_validation_batch (`bool`):
            Flag to check if current state is processing the last validation batch
        is_last_epoch (`bool`):
            Flag to check if current state is processing the last epoch.
        evaluations_done (`int`):
            Number of evaluations done.
        additional_metrics (`dict`):
            Additional metrics (e.g. accuracy, bleu, f1, etc).
        model_savings (`dict`):
            Bests model saving values.
        patience_left (`dict`):
            Patience left per model saving (in case it's implemented, otherwise values are set to -1).
        best_train_loss (`float`):
            Best training loss achieved.
        best_valid_loss (`float`):
            Best validation loss achieved.
        finished (`bool`, *optional*, defaults to `False`):
            Flag to identify if the process has already finished.
        num_checkpoints_made (`int`, *optional*, defaults to `0`):
            Number of checkpoints made.
    """

    batch_iteration: int = field(default=0)
    global_step: int = field(default=0)
    train_step: int = field(default=0)
    val_step: int = field(default=0)
    epoch: int = field(default=0)
    is_end_of_epoch: bool = field(default=False)
    is_last_training_batch: bool = field(default=False)
    is_last_validation_batch: bool = field(default=False)
    is_last_epoch: bool = field(default=False)

    evaluations_done: int = field(default=0)
    additional_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    patience_left: dict[str, int] = field(default_factory=dict)
    best_train_loss: float = field(default=float("inf"))
    finished: bool = field(default=False)
    num_checkpoints_made: int = field(default=0)

    def update(
        self,
        *,
        batch_iteration: Optional[int] = None,
        global_step: Optional[int] = None,
        train_step: Optional[int] = None,
        val_step: Optional[int] = None,
        epoch: Optional[int] = None,
        is_end_of_epoch: Optional[bool] = None,
        is_last_training_batch: Optional[bool] = None,
        is_last_validation_batch: Optional[bool] = None,
        is_last_epoch: Optional[bool] = None,
        evaluations_done: Optional[int] = None,
        additional_metrics: Optional[dict[str, dict[str, Any]]] = None,
        patience_left: Optional[dict[str, int]] = None,
        best_train_loss: Optional[float] = None,
        finished: Optional[bool] = None,
        num_checkpoints_made: Optional[int] = None,
        **kwargs: Any,
    ):
        # ignore positional arguments for safety
        updates = {
            "batch_iteration": batch_iteration,
            "global_step": global_step,
            "train_step": train_step,
            "val_step": val_step,
            "epoch": epoch,
            "is_end_of_epoch": is_end_of_epoch,
            "is_last_epoch": is_last_epoch,
            "is_last_training_batch": is_last_training_batch,
            "is_last_validation_batch": is_last_validation_batch,
            "evaluations_done": evaluations_done,
            "additional_metrics": additional_metrics,
            "patience_left": patience_left,
            "best_train_loss": best_train_loss,
            "finished": finished,
            "num_checkpoints_made": num_checkpoints_made,
        }

        for key, value in updates.items():
            if value is not None:
                setattr(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def from_dict(self, _dict: dict[str, Any]):
        self.update(**_dict)

    def save(self, to: str, _dict: Optional[dict[str, Any]] = None):
        state = self.to_dict() if _dict is None else _dict
        json.dump(state, open(to, "w"), indent=4, ensure_ascii=False)

    def load(self, path: str):
        _dict = json.load(open(path))
        self.from_dict(_dict)


class LossState:
    @torch.inference_mode()
    def __init__(
        self,
        accelerator: Accelerator,
        device: torch.device,
        log_every: int = -1,
        pin_memory: bool = True,
        include_per_batch: bool = True,
    ):
        self.accelerator = accelerator
        if device == torch.device("cpu") and pin_memory:
            pin_memory = False

        if log_every < 0:
            int_dtype = torch.int64  # value of -1 or lower means maximum dtype
        elif log_every < 2147483648:
            int_dtype = torch.int32
        else:
            int_dtype = torch.int64

        kwargs = {"device": device, "pin_memory": pin_memory}

        # Keep all tensors in the same device to avoid GPU -> CPU transfer.
        # Also, all tensors must be same data type to avoid internal convertions.
        self.num_batches = torch.tensor(0, dtype=int_dtype, **kwargs) if include_per_batch else None
        self.batch_loss = torch.tensor(0, dtype=torch.float32, **kwargs) if include_per_batch else None

        self.num_steps = torch.tensor(0, dtype=int_dtype, **kwargs)
        self.total_loss = torch.tensor(0, dtype=torch.float32, **kwargs)

        # create this extra tensor to avoid creating additional tensors to compute additions (~1.07x speedup)
        self._incrementor = torch.tensor(1, dtype=int_dtype, **kwargs)

        # This produces extra memory consumption in all devices, so per device
        # there are going to be (at maximum) 28 bytes in total allocated.

        self._include_per_batch = include_per_batch

    @torch.inference_mode()
    def get_batch_loss(self) -> float:
        if not self._include_per_batch:
            self.accelerator.end_training()
            raise RuntimeError("Batch loss calculation is not implemented. Use 'include_per_batch'.")

        batch_loss = self.batch_loss / self.num_batches
        batch_loss = self.accelerator.reduce(batch_loss, reduction="mean")  # GPU intercommunication
        self.num_batches.zero_()  # reset batches count
        self.batch_loss.zero_()  # reset batch loss tracker

        return batch_loss.item()  # CPU transfer

    @torch.inference_mode()
    def add_batch_loss(self, value: torch.Tensor):
        if not self._include_per_batch:
            self.accelerator.end_training()
            raise RuntimeError("Batch loss calculation is not implemented. Use 'include_per_batch'.")

        self.batch_loss.add_(value)
        self.num_batches.add_(self._incrementor)

    @torch.inference_mode()
    def get_total_loss(self) -> float:
        total_loss = self.total_loss / self.num_steps
        total_loss = self.accelerator.reduce(total_loss, reduction="mean")  # GPU intercommunication
        self.total_loss.zero_()
        self.num_steps.zero_()

        return total_loss.item()  # CPU transter

    @torch.inference_mode
    def add_total_loss(self, value: torch.Tensor):
        self.total_loss.add_(value)
        self.num_steps.add_(self._incrementor)

    @torch.inference_mode()
    def reset(self):
        if self._include_per_batch:
            self.num_batches.zero_()
            self.batch_loss.zero_()

        self.num_steps.zero_()
        self.total_loss.zero_()

    def save(self, path: str):
        _dict = {
            "steps": self.accelerator.gather(self.num_steps).cpu(),
            "total_loss": self.accelerator.gather(self.total_loss).cpu(),
        }

        if self._include_per_batch:
            _dict.update(
                {
                    "batches": self.accelerator.gather(self.num_batches).cpu(),
                    "batch_loss": self.accelerator.gather(self.batch_loss).cpu(),
                }
            )

        if MASTER_PROCESS:
            torch.save(_dict, path)

    @torch.inference_mode()
    def load(self, path: str):
        _dict = torch.load(path)
        with self.accelerator.split_between_processes(_dict) as inputs:
            if self._include_per_batch:
                self.num_batches.fill_(inputs["batches"].squeeze_())
                self.batch_loss.fill_(inputs["batch_loss"].squeeze_())

            self.num_steps.fill_(inputs["steps"].squeeze_())
            self.total_loss.fill_(inputs["total_loss"].squeeze_())
