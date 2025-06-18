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
import torch.nn.functional as F

from accmt import AcceleratorModule


class DummyClassificationModule(AcceleratorModule):
    def __init__(self):
        super().__init__()  # Initialize parent class
        self.model = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),  # 10 output classes (1-10)
        )

    def training_step(self, batch):
        x, y = batch
        output = self.model(x)
        loss = F.cross_entropy(output, y.argmax(dim=-1))

        return loss

    def validation_step(self, key, batch):
        x, y = batch
        output = self.model(x)
        loss = F.cross_entropy(output, y.argmax(dim=-1))

        return {
            "loss": loss,
            "accuracy": (output.argmax(dim=-1), y.argmax(dim=-1)),
        }
