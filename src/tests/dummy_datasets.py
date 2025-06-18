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

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    def __init__(self):
        self.dataset = [i / 10 for i in range(10)]
        self.labels = [i for i in range(10)]

    def __getitem__(self, idx):
        x = torch.tensor([self.dataset[idx]], dtype=torch.float32)
        y = F.one_hot(torch.tensor(self.labels[idx], dtype=torch.long), num_classes=10).float()

        return x, y

    def __len__(self):
        return len(self.dataset)
