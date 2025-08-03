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

import os


__version__ = "1.9.2.1"

ASYNC = bool(int(os.environ.get("ACCMT_ASYNC", 0)))
ASYNC_HASH = os.environ.get("ACCMT_HASH", None)
ASYNC_TRAIN_GROUP = bool(int(os.environ.get("ACCMT_TRAIN_GROUP", 0)))
DIST_HASH = ASYNC_HASH
IS_CPU = bool(int(os.environ.get("ACCMT_CPU", 0)))
IS_GPU = not IS_CPU
DEBUG_MODE = int(os.environ.get("ACCMT_DEBUG_MODE", 0))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
RANK = int(os.getenv("RANK", 0))
MASTER_PROCESS = RANK == 0
LAST_PROCESS = RANK == WORLD_SIZE - 1
