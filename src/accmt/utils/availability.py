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

import torch


def is_tf32_supported() -> bool:
    """
    Check if TensorFloat32 is supported. Implementation is identical to `torch.cuda.is_tf32_supported`
    in `torch` library (v2.6.0+).

    Returns:
        `bool`: Whether TensorFloat32 is supported.
    """
    # Check for ROCm.  If true, return false, since PyTorch does not currently support
    # tf32 on ROCm.
    if torch.version.hip:
        return False

    # Otherwise, tf32 is supported on CUDA platforms that natively (i.e. no emulation)
    # support bfloat16.
    return torch.cuda.is_bf16_supported(including_emulation=False)


def _is_module_available(module_name: str) -> bool:
    """
    Check if a Python module is available for import.

    Args:
        module_name (str): Name of the module to check.

    Returns:
        bool: True if the module can be imported, False otherwise.
    """
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def is_transformers_available() -> bool:
    return _is_module_available("transformers")


def is_pandas_available() -> bool:
    return _is_module_available("pandas")


def is_deepspeed_available() -> bool:
    return _is_module_available("deepspeed")
