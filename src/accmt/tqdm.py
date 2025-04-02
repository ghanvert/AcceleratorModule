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

from tqdm.auto import tqdm as _tqdm

from .utility import MASTER_PROCESS


class tqdm(_tqdm):
    """Wrapper around tqdm to only run on main process and have precision of seconds."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, disable=not MASTER_PROCESS, **kwargs)

    @property
    def format_dict(self):
        d = super().format_dict
        rate_s = "{:.3f}".format(1 / d["rate"]) if d["rate"] else "?"
        d.update(rate_s=(rate_s + " s"))
        return d
