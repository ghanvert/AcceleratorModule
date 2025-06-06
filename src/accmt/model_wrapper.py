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

from torch.nn.parallel import DistributedDataParallel


class _DistributedDataParallel:
    """
    Wrapper around DDP wrapper to solve issues detected with DDP when user calls a function
    not present in wrapper like 'generate'.
    """

    def __init__(self, model: DistributedDataParallel):
        self._model = model

    def __getattr__(self, name):
        if hasattr(self._model, name):
            return getattr(self._model, name)
        else:
            return getattr(self._model.module, name)

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)
