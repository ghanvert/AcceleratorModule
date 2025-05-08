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


class DummyModel(nn.Module):
    def __init__(self, input_size: int, inner_size: int, output_size: int):
        super().__init__()
        self.input_layer = nn.Linear(input_size, inner_size)
        self.hidden_layer = nn.Linear(inner_size, inner_size)
        self.output_layer = nn.Linear(inner_size, output_size)

    def forward(self, x):
        x = self.input_layer(x)
        x = F.tanh(x)
        x = self.hidden_layer(x)
        x = F.sigmoid(x)
        x = self.output_layer(x)

        return x
