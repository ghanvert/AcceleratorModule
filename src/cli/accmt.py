#!/usr/bin/env python
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

from .launch import launch
from .parser import get_parser
from .utils import (
    DEBUG_LEVEL_INFO,
    generate_hps,
    get_python_cmd,
    remove_compiled_prefix,
    show_strategies,
)


def main():
    parser, args = get_parser()

    if args.command is None:
        parser.print_help()
        exit(0)

    import torch

    if ("debug" in args.command or "launch" in args.command) and args.command != "debug-levels":
        launch(args)
    elif args.command == "get":
        assert args.out is not None, "You must specify an output directory ('--out')."
        assert hasattr(torch, args.dtype), f"'{args.dtype}' not supported in PyTorch."
        CHKPT_BASE_DIRECTORY = f"{args.checkpoint}/checkpoint"
        checkpoint_dir = CHKPT_BASE_DIRECTORY if os.path.exists(CHKPT_BASE_DIRECTORY) else args.get
        files = os.listdir(checkpoint_dir)

        python_cmd = get_python_cmd()
        os.makedirs(args.out, exist_ok=True)
        if "status.json" in os.listdir(args.checkpoint):
            shutil.copy(f"{args.checkpoint}/status.json", args.out)

        state_dict_file = f"{args.out}/pytorch_model.pt"

        if "zero_to_fp32.py" in files:  # check for DeepSpeed
            print("Converting Zero to float32 parameters...")
            exit_code = os.system(f"{python_cmd} {checkpoint_dir}/zero_to_fp32.py {checkpoint_dir} {state_dict_file}")
            if exit_code != 0:
                raise RuntimeError("Something went wrong when converting Zero to float32.")
        elif "pytorch_model_fsdp_0" in files:  # check for FSDP
            # using Accelerate's approach for now, and only checking for one node
            exit_code = os.system(f"accelerate merge-weights {checkpoint_dir}/pytorch_model_fsdp_0 {args.out}")
            if exit_code != 0:
                raise RuntimeError("Something went wrong when merging weights from FSDP.")

            shutil.copy(f"{checkpoint_dir}/pytorch_model.bin", f"{args.out}/pytorch_model.pt")
        else:  # check for DDP
            shutil.copy(f"{checkpoint_dir}/pytorch_model.bin", f"{args.out}/pytorch_model.pt")

        state_dict = torch.load(state_dict_file, map_location="cpu", weights_only=True)
        state_dict = remove_compiled_prefix(state_dict)

        _dtype_str = f" and converting to dtype {args.dtype}" if args.dtype is not None else ""
        print(f"Setting 'requires_grad' to False{_dtype_str}...")
        for key in state_dict.keys():
            state_dict[key].requires_grad = False
            if args.dtype is not None:
                state_dict[key] = state_dict[key].to(getattr(torch, args.dtype))

        torch.save(state_dict, state_dict_file)
        print(f"Model directory saved to '{args.out}'.")
    elif args.command == "strats":
        if args.ddp:
            show_strategies(filter="ddp")
        elif args.fsdp:
            show_strategies(filter="fsdp")
        elif args.deepspeed:
            show_strategies(filter="deepspeed")
        else:
            show_strategies()
    elif args.command == "example":
        generate_hps()
        print("'hps_example.yaml' generated.")
    elif args.command == "debug-levels":
        _default_str = " (default)"
        if args.level is None:
            for level, info in DEBUG_LEVEL_INFO.items():
                print(f"  Level {level}{_default_str if level == 3 else ''}: {info}")
        else:
            if args.level in DEBUG_LEVEL_INFO:
                print(f"  Level {args.level}: {DEBUG_LEVEL_INFO[args.level]}")
            else:
                print(f"Level {args.level} is not valid. Debug mode levels are: {list(DEBUG_LEVEL_INFO.keys())}")


if __name__ == "__main__":
    main()
