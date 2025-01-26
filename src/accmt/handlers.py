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

from dataclasses import dataclass


@dataclass
class Handler:
    """
    Properties:
        `KEYBOARD`: Handles `KeyboardInterrupt` exceptions (CTRL + C).
        `CUDA_OUT_OF_MEMORY`: Handles cuda out of memory errors derived from `RuntimeError` exceptions.
        `ANY`: Handles any other exception not considered in these properties.
        `ALL`: Handles all possible exceptions.
    """

    KEYBOARD = 1
    CUDA_OUT_OF_MEMORY = 2
    ANY = 3
    ALL = 4
