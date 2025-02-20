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

import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from .. import accelerator


# TODO Work in progress...


class Pipeline:
    def __init__(self):
        # TODO this should handle cases for DeepSpeed for multiple models
        self._model_count = 0

    def prepare(self):
        for key, attr in self.__dict__.items():
            cls_type = type(attr)
            if cls_type is nn.Module:
                self.prepare_model(key, attr)
            elif cls_type is DataLoader:
                self.prepare_dataloader(key, attr)
            elif cls_type is LRScheduler:
                self.prepare_scheduler(key, attr)

    def prepare_model(self, key: str, model: nn.Module):
        setattr(self, key, accelerator.prepare_model(model))
        self._model_count += 1

    def prepare_dataloader(self, key: str, dataloader: DataLoader):
        setattr(self, key, accelerator.prepare_data_loader(dataloader))

    def prepare_scheduler(self, key: str, scheduler: LRScheduler):
        setattr(self, key, accelerator.prepare_scheduler(scheduler))


class TrainingPipeline(Pipeline):
    def __init__(self):
        self.epochs = -1
        self.train_dataloader = ...
        self.val_dataloader = ...

    def loop(self):
        for epoch in self.epoch_iterator():
            for batch in self.train_dataloader:
                self.training_step(batch)

            for batch in self.val_dataloader:
                self.validation_step(batch)

    def epoch_iterator(self):
        """Epoch iterator logic"""

    def save_checkpoint(self):
        """Checkpoint logic"""

    def save_best_model(self):
        """Save best model logic"""

    def log(self):
        """Logging logic"""
