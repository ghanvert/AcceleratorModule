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

import subprocess
from typing import Optional

from .globals import MASTER_PROCESS


def _run_claude_cmd(*, prompt: str, agent: Optional[str] = None, skip_permissions: bool = False):
    if not MASTER_PROCESS:
        return

    cmd = ["claude", "-p", prompt]

    if agent is not None:
        cmd.extend(["--agent", agent])

    if skip_permissions:
        cmd.append("--dangerously-skip-permissions")

    subprocess.Popen(cmd, stdout=subprocess.DEVNULL)


def run_claude(*, prompt: str, agent: Optional[str] = None, skip_permissions: bool = False):
    """
    Run Claude on a given prompt. Only runs on the master process.

    Args:
        prompt (`str`):
            The prompt to give to Claude.
        agent (`str`, *optional*, defaults to `None`):
            The name of the Claude agent to run. If `None`, runs the default Claude model.
        skip_permissions (`bool`, *optional*, defaults to `False`):
            Whether to skip the permissions check. By default, Claude is only
    """
    _run_claude_cmd(prompt=prompt, agent=agent, skip_permissions=skip_permissions)
