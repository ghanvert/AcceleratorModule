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

import hashlib
import os
import signal
import subprocess
import sys
from datetime import datetime

import torch

from .utils import configs, get_cmd_as_list, get_free_gpus, modify_config_file


def launch(args):
    cpu = False
    _async = False

    current_time = datetime.now().isoformat()
    hash_object = hashlib.sha256(current_time.encode())
    hash_hex = hash_object.hexdigest()
    os.environ["ACCMT_HASH"] = hash_hex

    if "debug" in args.command:
        os.environ["ACCMT_DEBUG_MODE"] = str(args.level)

    if args.command in {"alaunch", "async-launch", "adebug", "async-debug"}:
        os.environ["ACCMT_ASYNC"] = "1"
        _async = True

    if args.cpu:
        os.environ["ACCMT_CPU"] = "1"
        cpu = True

    gpus = args.gpus.lower()
    strat = args.strat
    file = args.file
    extra_args = " ".join(args.extra_args)

    if "." in strat:
        accelerate_config_file = strat
    else:
        accelerate_config_file = configs[strat]

    if not cpu:
        if not torch.cuda.is_available():
            raise ImportError("Could not run CLI: CUDA is not available on your PyTorch installation.")

        NUM_DEVICES = torch.cuda.device_count()

        gpu_indices = ""
        if gpus == "available":
            gpu_indices = ",".join(get_free_gpus(NUM_DEVICES))
        elif gpus == "all":
            gpu_indices = ",".join(str(i) for i in range(NUM_DEVICES))
        else:
            gpu_indices = gpus.removeprefix(",").removesuffix(",")

        if gpu_indices == "":
            raise RuntimeError(
                "Could not get GPU indices. If you're using 'available' in 'gpus' "
                "parameter, make sure there is at least one GPU free of memory."
            )

        if not _async and args.N != "0":
            if ":" in args.N:
                _slice = slice(*map(lambda x: int(x.strip()) if x.strip() else None, args.N.split(":")))
                gpu_indices = ",".join([str(i) for i in range(NUM_DEVICES)][_slice])
            else:
                gpu_indices = ",".join(str(i) for i in range(int(args.N)))

        # TODO: For now, we need to find a way to collect processes that are running on certain GPUs to verify if they're free to use.
        # if not args.ignore_warnings:
        #    gpu_indices_list = [int(idx) for idx in gpu_indices.split(",")]
        #    device_indices_in_use = []
        #    for idx in gpu_indices_list:
        #        if cuda_device_in_use(idx):
        #            device_indices_in_use.append(idx)
        #
        #    if len(device_indices_in_use) > 0:
        #        raise RuntimeError(
        #            f"The following CUDA devices are in use: {device_indices_in_use}."
        #             "You can ignore this warning via '--ignore-warnings'."
        #        )

        num_processes = len(gpu_indices.split(","))
    else:
        if args.N == "0":
            raise RuntimeError("When running on CPU, '-N' must specify the number of processes to run.")

        if _async:
            raise NotImplementedError("Asynchronous evaluations are not supported for CPU.")

        num_processes = int(args.N)

    port, _ = modify_config_file(accelerate_config_file, num_processes)

    if args.command == "example":
        print(f"accelerate launch --config_file={accelerate_config_file} {file} {extra_args}")
        return

    if not cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_indices
    cmd = f"accelerate launch --config_file={accelerate_config_file} {file} {extra_args}"

    omp_num_threads = os.cpu_count() // num_processes
    if not _async:
        if not cpu:
            os.environ["OMP_NUM_THREADS"] = str(omp_num_threads)

        os.system(cmd)
    else:
        train_group = os.environ.copy()
        train_group["ACCMT_TRAIN_GROUP"] = "1"
        train_group["OMP_NUM_THREADS"] = str(omp_num_threads)
        cmd = get_cmd_as_list(cmd)
        try:
            process1 = subprocess.Popen(cmd, env=train_group, start_new_session=True)
            process2 = None

            if _async:
                eval_group = os.environ.copy()
                gpu_indices = args.evaluation_device_indices.removeprefix(",").removesuffix(",")
                if not cpu:
                    eval_group["CUDA_VISIBLE_DEVICES"] = gpu_indices
                    omp_num_threads = os.cpu_count() // len(gpu_indices.split(","))
                    eval_group["OMP_NUM_THREADS"] = str(omp_num_threads)

                num_processes = len(gpu_indices.split(","))

                _, _accelerate_config_file = modify_config_file(
                    accelerate_config_file, num_processes, port=port + 1, copy=True
                )
                async_cmd = f"accelerate launch --config_file={_accelerate_config_file} {file} {extra_args}"

                async_cmd = get_cmd_as_list(async_cmd)
                process2 = subprocess.Popen(async_cmd, env=eval_group, start_new_session=True)

                process2.wait()
                process1.wait()
            else:
                process1.wait()
        except KeyboardInterrupt:
            print("\nTerminating subprocesses...")
            # Send SIGTERM to entire process groups
            processes = [process1]
            if process2 is not None:
                processes.append(process2)
            for p in processes:
                if p and p.poll() is None:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            sys.exit(1)
