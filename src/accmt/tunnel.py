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

import json
import os
import threading
import time
from multiprocessing import shared_memory
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator

from .utility import MASTER_PROCESS


ASYNC_HASH = os.environ.get("ACCMT_HASH", None)


class ModelTunnel:
    def __init__(self, name: str):
        # ignore memory leak warnings from 'resource_tracker' since 'close()' function
        # will only be called at the end of the program.
        os.environ["PYTHONWARNINGS"] = "ignore:resource_tracker:UserWarning"
        self.name = name
        self.shm: shared_memory.SharedMemory = None
        self.shm_array = None

    def init(self, model: nn.Module):
        if MASTER_PROCESS:
            if self.shm is not None:
                raise RuntimeError("Cannot re-create tunnel.")

            params = [p.data.view(-1).cpu().float().numpy() for p in model.state_dict().values()]
            flat_params = np.concatenate(params)
            self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=flat_params.nbytes)
            self.write(model, flat_params)

    def write(self, model: nn.Module, data: Optional[np.ndarray] = None):
        if MASTER_PROCESS:
            if data is None:
                data = [p.data.view(-1).cpu().float().numpy() for p in model.state_dict().values()]
                data = np.concatenate(data)

            self.shm_array = np.ndarray(data.shape, dtype=data.dtype, buffer=self.shm.buf)
            self.shm_array[:] = data[:]

    def write_state_dict(self, state_dict: dict[str, torch.Tensor], non_blocking: bool = False):
        def _write_state_dict():
            data = [p.data.view(-1).cpu().float().numpy() for p in state_dict.values()]
            data = np.concatenate(data)

            self.shm_array = np.ndarray(data.shape, dtype=data.dtype, buffer=self.shm.buf)
            self.shm_array[:] = data[:]

        if MASTER_PROCESS:
            if non_blocking:
                thread = threading.Thread(target=_write_state_dict)
                thread.start()
            else:
                _write_state_dict()

    def read(self, model: nn.Module):
        if MASTER_PROCESS:
            if self.shm_array is None:
                self.shm = shared_memory.SharedMemory(name=self.name)

            device = next(model.parameters()).device
            # calculate total number of weights
            param_shapes = [p.shape for p in model.state_dict().values()]
            num_weights = sum(np.prod(shape) for shape in param_shapes)

            # read flat array from shared memory
            self.shm_array = np.ndarray((num_weights,), dtype=np.float32, buffer=self.shm.buf)

            # copy values back into model parameters
            offset = 0
            for param in model.parameters():
                numel = param.numel()
                param.data.copy_(
                    (torch.from_numpy(self.shm_array[offset : offset + numel]).view(param.shape).pin_memory()),
                    non_blocking=True,
                )
                offset += numel

            if device != torch.device("cpu"):
                torch.cuda.synchronize(0)

            return model

    def close(self):
        if MASTER_PROCESS:
            try:
                if self.shm_array is not None:
                    del self.shm_array

                if self.shm is not None:
                    self.shm.unlink()
                    self.shm.close()
            except FileNotFoundError:
                # SHM already destroyed
                pass


class AsyncState:
    def __init__(self, model_path: str):
        self.path = os.path.join(model_path, f"ASYNC_{ASYNC_HASH}.json")
        self.state = self._set_initial_state()

    def init(self):
        self.update()

    def update(
        self,
        train_finished: Optional[bool] = None,
        evaluation_finished: Optional[bool] = None,
        evaluations_in_queue: Optional[int] = None,
        tunnel_ready: Optional[bool] = None,
        sync_requested: Optional[bool] = None,
    ):
        self._update(train_finished, evaluation_finished, evaluations_in_queue, tunnel_ready, sync_requested)

        if MASTER_PROCESS:
            while True:
                try:
                    json.dump(self.state, open(self.path, "w"))
                    break
                except PermissionError:  # file being used
                    self._update(
                        train_finished, evaluation_finished, evaluations_in_queue, tunnel_ready, sync_requested
                    )
                    time.sleep(0.5)
                    continue

    def _update(self, train_finished, evaluation_finished, evaluations_in_queue, tunnel_ready, sync_requested):
        self._read_from_disk()
        if train_finished is not None:
            self.state["train_finished"] = train_finished
        if evaluation_finished is not None:
            self.state["evaluation_finished"] = evaluation_finished
        if evaluations_in_queue is not None:
            self.state["evaluations_in_queue"] += evaluations_in_queue
        if tunnel_ready is not None:
            self.state["tunnel_ready"] = tunnel_ready
        if sync_requested is not None:
            self.state["sync_requested"] = sync_requested

    def _read_from_disk(self):
        if os.path.exists(self.path):
            while True:
                try:
                    self.state = json.load(open(self.path))
                    return
                except json.JSONDecodeError:
                    # this will raise most of the times due to file still being written
                    time.sleep(0.5)
                    continue
                except PermissionError:
                    # this will raise in rare cases
                    time.sleep(0.5)
                    continue

        self.state = self._set_initial_state()

    def _set_initial_state(self) -> dict[str, Any]:
        return {
            "train_finished": False,
            "evaluation_finished": False,
            "evaluations_in_queue": 0,
            "tunnel_ready": False,
            "sync_requested": False,
        }

    @property
    def train_finished(self) -> bool:
        self._read_from_disk()
        return self.state["train_finished"]

    @property
    def evaluation_finished(self) -> bool:
        self._read_from_disk()
        return self.state["evaluation_finished"]

    @property
    def evaluations_in_queue(self) -> int:
        self._read_from_disk()
        return self.state["evaluations_in_queue"]

    @property
    def tunnel_ready(self) -> bool:
        self._read_from_disk()
        return self.state["tunnel_ready"]

    @property
    def sync_requested(self) -> bool:
        self._read_from_disk()
        return self.state["sync_requested"]

    def request_sync(self):
        self.update(sync_requested=True)

    def wait_for_sync(self, delay: float = 0.1):
        while not self.sync_requested:
            time.sleep(delay)

        self.update(sync_requested=False)

    def wait_for_tunnel(self, delay: float = 0.1):
        while not self.tunnel_ready:
            time.sleep(delay)


class AsyncDiskQueue:
    def __init__(self, model_path, accelerator: Accelerator):
        self.path = os.path.join(model_path, "async_queue")
        self.accelerator = accelerator
        if MASTER_PROCESS:
            os.makedirs(self.path, exist_ok=True)

    def get_queue(self) -> list[str]:
        return [os.path.join(self.path, f) for f in os.listdir(self.path)]

    def is_empty(self) -> bool:
        return len(self.get_queue()) == 0

    @property
    def size(self):
        return len(os.listdir(self.path))

    def front(self):
        dir_list = os.listdir(self.path)
        if len(dir_list) == 0:
            return None

        return sorted(dir_list, key=lambda x: int(x))[0]

    def back(self):
        dir_list = os.listdir(self.path)
        if len(dir_list) == 0:
            return None

        return sorted(dir_list, key=lambda x: int(x))[-1]

    def enqueue(self, unwrapped_model: nn.Module):
        if self.size == 0:
            last_added_model_id = "0"
        else:
            last_added_model_id = str(int(self.back()) + 1)

        path = os.path.join(self.path, last_added_model_id)
        state_dict = unwrapped_model.state_dict()
        if MASTER_PROCESS:
            os.makedirs(path, exist_ok=True)
            pt_state_dict = os.path.join(path, "pytorch_model.pt")
            self.accelerator.save(state_dict, pt_state_dict)

    def dequeue(self) -> dict:
        next_model_id = self.front()
        path = os.path.join(self.path, next_model_id)
        state_dict_path = os.path.join(path, "pytorch_model.pt")
        state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)

        return state_dict
