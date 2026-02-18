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

from argparse import REMAINDER, ArgumentParser, Namespace


def add_launch_arguments(parser: ArgumentParser, _async: bool = False):
    parser.add_argument(
        "--gpus",
        "-n",
        default="all",
        type=str,
        required=_async,
        help="Number or GPU indices to use (e.g. -n=0,1,4,5 | -n=all | -n=available).",
    )
    if not _async:
        parser.add_argument(
            "-N",
            default="0",
            type=str,
            required=False,
            help="Number of GPUs to use. This does not consider GPU indices by default, although you can represent "
            "a Python slice. (e.g. '2:', which means from index 2 to the last GPU index, or "
            "'3:8', which means from index 3 to index 7, or lastly ':4', which means indices 0 to 3 or a total of 4 gpus).",
        )
    parser.add_argument(
        "--strat",
        type=str,
        required=False,
        default="ddp",
        help="Parallelism strategy to apply or config file path. See 'accmt strats'.",
    )
    parser.add_argument("--cpu", action="store_true", help="Destinate this process to CPU.")
    parser.add_argument("--debug-timings", action="store_true", help="Debug timings.")

    # TODO: For now, we need to find a way to collect processes that are running on certain GPUs to verify if they're free to use.
    # parser.add_argument("--ignore-warnings", action="store_true", help="Ignore warnings (launch independent if GPUs are being used).")

    if _async:
        parser.add_argument(
            "--evaluation-device-indices",
            "-e",
            type=str,
            required=True,
            help="Number or GPU indices to use for evaluation (e.g. -n=0,1,4,5).",
        )
    parser.add_argument("file", type=str, help="File to run training.")
    parser.add_argument("extra_args", nargs=REMAINDER)


def get_parser() -> tuple[ArgumentParser, Namespace]:
    parser = ArgumentParser(description="AcceleratorModule CLI to run train processes on top of 🤗 Accelerate.")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Run distributed training
    launch_parsers, debug_parsers = [], []

    launch_parsers.append(subparsers.add_parser("launch", help="Launch distributed training processes."))
    launch_parsers.append(
        subparsers.add_parser("alaunch", help="Launch distributed training processes with asynchronous evaluation.")
    )
    launch_parsers.append(
        subparsers.add_parser(
            "async-launch", help="Launch distributed training processes with asynchronous evaluation."
        )
    )
    debug_parsers.append(subparsers.add_parser("debug", help="Launch distributed training processes in debug mode."))
    debug_parsers.append(
        subparsers.add_parser(
            "adebug", help="Launch distributed training processes with asynchronous evaluation in debug mode."
        )
    )
    debug_parsers.append(
        subparsers.add_parser(
            "async-debug", help="Launch distributed training processes with asynchronous evaluation in debug mode."
        )
    )

    for launch_parser in launch_parsers:
        _async = launch_parser.prog.split(" ")[-1] in {"alaunch", "async-launch"}
        add_launch_arguments(launch_parser, _async=_async)

    for debug_parser in debug_parsers:
        _async = debug_parser.prog.split(" ")[-1] in {"adebug", "async-debug"}
        add_launch_arguments(debug_parser, _async=_async)
        debug_parser.add_argument(
            "--level",
            "-L",
            "-l",
            type=int,
            default=3,
            required=False,
            help="Debug mode level (default is 3). See more details using 'accmt debug-levels'.",
        )

    # Get model from checkpoint
    get_parser = subparsers.add_parser("get", help="Get model from a checkpoint directory.")
    get_parser.add_argument("checkpoint", type=str, help="Checkpoint directory.")
    get_parser.add_argument("--out", "-O", "-o", required=True, type=str, help="Output directory path name.")
    get_parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help=(
            "Data type of model parameters. Available options are all those from PyTorch ('float32', 'float16', etc)."
        ),
    )

    # Strats
    strats_parser = subparsers.add_parser("strats", help="Available strategies.")
    strats_parser.add_argument(
        "--ddp", action="store_true", help="Only show DistributedDataParallel (DDP) strategies."
    )
    strats_parser.add_argument(
        "--fsdp", action="store_true", help="Only show FullyShardedDataParallel (FSDP) strategies."
    )
    strats_parser.add_argument("--deepspeed", action="store_true", help="Only show DeepSpeed strategies.")

    # Debug levels
    debug_levels_parser = subparsers.add_parser("debug-levels", help="Available debug levels.")
    debug_levels_parser.add_argument(
        "--level", "-L", "-l", type=int, required=False, help="See details about a specific debug mode level."
    )

    # Generate example
    subparsers.add_parser("example", help="Generate example file.")

    return parser, parser.parse_args()
