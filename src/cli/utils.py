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
import shutil
import socket

import yaml


configs = {}
_directory = os.path.dirname(__file__)
for file in os.listdir(f"{_directory}/config"):
    key = file.split(".")[0]
    configs[key] = f"{_directory}/config/{file}"


def get_free_gpus(num_devices: int) -> list[str]:
    import torch

    GB = 1024**3

    devices = []
    for i in range(num_devices):
        mem_total, mem_alloc, mem_resvd = get_cuda_device_memory(i)
        if mem_total > 0:
            device_name = torch.cuda.get_device_name(i)
            print("------------------------------------------------")
            print("GPU in use:")
            print(f"Name: {device_name}")
            print(f"Memory allocated: {mem_alloc / GB} GB")
            print(f"Memory reserved: {mem_resvd / GB} GB")
            print("ACCMT will not use this GPU during training.")
            print("------------------------------------------------")

            continue

        devices.append(str(i))

    return devices


def check_port_available(port: int, host="127.0.0.1"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        return result != 0


def modify_config_file(path: str, num_gpus: int, port: int = 29500, copy: bool = False) -> tuple[int, str]:
    data = yaml.safe_load(open(path))

    if port != -1:
        _port = port
        port = port if check_port_available(port) else 0
        if port == 0:
            for current_port in range(_port + 1, 65536):
                if check_port_available(current_port):
                    port = current_port
                    break

            if port == 0:  # if 29500 to 65535 is not available
                for current_port in range(1, _port):
                    if check_port_available(current_port):
                        port = current_port
                        break

            if port == 0:
                raise RuntimeError("There are no ports available in your system.")
    else:
        port = 0  # accelerate will automatically check for an available port

    prev_main_process_port = data["main_process_port"] if "main_process_port" in data else -1
    prev_num_processes = data["num_processes"]

    if prev_main_process_port == port and prev_num_processes == num_gpus:
        return port, path  # skip write process

    data["main_process_port"] = port
    data["num_processes"] = num_gpus

    if copy:
        base_dir, filename = os.path.split(path)
        filename = f"_{filename}"
        path = os.path.join(base_dir, filename)

    with open(path, "w") as f:
        yaml.safe_dump(data, f)

    return port, path


def get_python_cmd():
    if shutil.which("python") is not None:
        return "python"
    else:
        return "python3"


def remove_compiled_prefix(state_dict):
    compiled = "_orig_mod" in list(state_dict.keys())[0]
    if not compiled:
        return state_dict

    t = type(state_dict)
    return t({k.removeprefix("_orig_mod."): v for k, v in state_dict.items()})


def show_strategies(filter: str = None):
    if filter is None:
        filter = ""
    for strat in configs.keys():
        if filter in strat:
            print(f"\t{strat}")

    exit(1)


def generate_hps():
    directory = os.path.dirname(__file__)
    shutil.copy(f"{directory}/example/hps_example.yaml", ".")


def get_cuda_device_memory(device: int | str) -> tuple[float, float, float]:
    import torch

    device = f"cuda:{device}" if isinstance(device, int) else device

    mem_alloc = torch.cuda.memory_allocated(device)
    mem_resvd = torch.cuda.memory_reserved(device)
    mem_total = mem_alloc + mem_resvd

    return mem_total, mem_alloc, mem_resvd


def cuda_device_in_use(device: int | str):
    mem_total, _, _ = get_cuda_device_memory(device)
    return mem_total > 0


def remove_first_line_in_file(file: str):
    with open(file, "r+") as f:
        lines = f.readlines()
        f.seek(0)
        f.writelines(lines[1:])
        f.truncate()


def get_cmd_as_list(cmd: str) -> list[str]:
    cmd = cmd.split(" ")
    cmd = [c for c in cmd if c not in {"", " "}]

    return cmd


DEBUG_LEVEL_INFO = {
    1: "Disables logging (MLFlow, Tensorboard, etc).",
    2: "Disables model and teacher compilation.",
    3: "Disables model saving, checkpointing and resuming (no folders will be created).",
    4: "Force 'eval_when_start' (in Trainer) to False.",
    5: "Disables any evaluation.",
}
