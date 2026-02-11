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

from .utils.globals import WORLD_SIZE


class CustomBatchSampler:
    """
    Custom batch sampler for dataset. When inherited, no data sharding will be performed automatically.
    The user is responsible for implementing the data sharding logic.

    `set_epoch` can be added as a method to the class to set the epoch and define RNG strategies.

    Example:
    ```python
    class CustomBatchSampler(CustomBatchSampler):
        def __init__(self, dataset, batch_size):
            super().__init__()
            self.dataset = dataset
            # there is an internal `batch_size` attribute, so we use `_batch_size` instead to
            # avoid any errors that might occur.
            self._batch_size = batch_size
            self.epoch = 0

        def __iter__(self):
            for i in range(0, len(self.dataset), self._batch_size):
                yield [idx for idx in range(i, i + self._batch_size)]

        def set_epoch(self, epoch):
            # you can use `self.epoch` to define RNG strategies.
            self.epoch = epoch
    ```
    """

    def __init__(self):
        self._custom_batch_sampler = True  # flag for trainer to ignore the data sharding
        self.batch_size = WORLD_SIZE  # avoid errors when `even_batches` in accelerator is enabled
