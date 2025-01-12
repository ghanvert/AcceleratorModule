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

import torch.nn as nn
from dummy_dataset import DummyDataset

from accmt import AcceleratorModule, HyperParameters, Trainer

from .dummy_model import DummyModel


class DummyModule(AcceleratorModule):
    def __init__(self):
        self.model = DummyModel(input_size=2, inner_size=5, output_size=3)
        self.criterion = nn.CrossEntropyLoss()

    def step(self, batch):
        x, y = batch
        x = self.model(x)

        loss = self.criterion(x, y)

        return loss


module = DummyModule()

train_dataset = DummyDataset()

trainer = Trainer(hps_config=HyperParameters(epochs=2, batch_size=2), model_path="dummy_model")
trainer.fit(module, train_dataset)
