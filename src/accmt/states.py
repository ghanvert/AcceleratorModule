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

from typing import Any, Optional, Union

import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader


class TrainingState:
    def __init__(
        self,
        model: Union[nn.Module, Any],
        wrapped_model: Union[nn.Module, Any],
        teacher: Union[nn.Module, Any],
        wrapped_teacher: Union[nn.Module, Any],
        optimizer: Optimizer,
        wrapped_optimizer: Union[Optimizer, Any],
        train_dataloader: DataLoader,
        wrapped_train_dataloader: Union[DataLoader, Any],
        val_dataloader: Optional[DataLoader] = None,
        wrapped_val_dataloader: Optional[Union[DataLoader, Any]] = None,
        scheduler: Optional[LRScheduler] = None,
        wrapped_scheduler: Optional[Union[LRScheduler, Any]] = None,
    ):
        self.model = model
        self.wrapped_model = wrapped_model
        self.teacher = teacher
        self.wrapped_teacher = wrapped_teacher
        self.optimizer = optimizer
        self.wrapped_optimizer = wrapped_optimizer
        self.train_dataloader = train_dataloader
        self.wrapped_train_dataloader = wrapped_train_dataloader
        self.val_dataloader = val_dataloader
        self.wrapped_val_dataloader = wrapped_val_dataloader
        self.scheduler = scheduler
        self.wrapped_scheduler = wrapped_scheduler
