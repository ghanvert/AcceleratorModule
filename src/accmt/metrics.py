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

from collections import defaultdict
from typing import Any, Optional, Union

import numpy as np
import torch
from typing_extensions import override


_available_comparators = ["<", "<=", ">", ">=", "=="]


class Metric:
    """Compute metrics on main process."""

    def __init__(
        self,
        name: str,
        greater_is_better: bool = True,
        main_metric: Optional[str] = None,
        do_checks: bool = True,
        cast: Optional[Union[torch.dtype, str]] = torch.float32,
    ):
        """
        Set a module to compute metrics. All computations are done in main process.

        Args:
            name (`str`):
                Metric's module name.
            greater_is_better (`bool`, *optional*, defaults to `True`):
                Specify if the main metric is better when is greater.
            main_metric (`str`, *optional*, defaults to `None`):
                Determine which is the main metric key in your compute output. By default, main metric key will be
                equal to the 'name' parameter.
            do_checks (`bool`, *optional*, defaults to `True`):
                Enable shape checks when appending metrics. This can be disabled for small speed improvements.
            cast (`dtype` or `str`, *optional*, defaults to `torch.float32`):
                Cast all floating point tensors to the desired `dtype`. If `None`, no upcasting will be done.
        """
        self.name = name
        comparator = ">=" if greater_is_better else "<="
        assert comparator in _available_comparators, f"Available options for comparator are: {_available_comparators}"
        self.greater_is_better = greater_is_better
        self.comparator = comparator
        self.main_metric = main_metric if main_metric is not None else name

        # Lists of every argument, where every argument is also a list of tensors (or dictionary). Example:
        #   [[tensor, tensor, tensor], [tensor, tensor, tensor], ...], {"x": [tensor, tensor, tensor], "y": ...}
        #   argument1                  argument2                       arguments...
        self.arguments = []

        from . import accelerator

        self.accelerator = accelerator
        self.do_checks = do_checks
        self._parallel = False
        if isinstance(cast, str):
            cast = getattr(torch, cast)
        self.cast = cast

    @override
    def compute(self, *args: Union[torch.Tensor, dict[Any, torch.Tensor]]) -> dict:
        """
        Compute metrics with the given arguments. This function returns a dictionary
        containing the main metric value and others.

        Example:
            ```
            def compute(self, predictions, references):
                # logic of how to calculate metrics here...

                return {
                    "accuracy": 0.85, # <-- this one is the main value
                    "f1": 0.89
                }
            ```

        NOTE: In the previous example, the main metric is 'accuracy', and its value is gonna be used along with
        'comparator' to compare if the metric is the best or not. By default, main metric is set to the name of
        the metric itself. You can change this behaviour with 'main_metric' on class initialization.
        """

    def _convert_to_python_values(self, dictionary: dict[str, Any]):
        for k, v in dictionary.items():
            if isinstance(v, (torch.Tensor, np.ndarray)):
                dictionary[k] = v.item()
            elif isinstance(v, dict):
                dictionary[k] = self._convert_to_python_values(v)
            elif not isinstance(v, (float, int)):
                raise ValueError(
                    f"Value in metric's dict does not accept {type(v)}, only "
                    f"`float`, `int`, `torch.Tensor` (torch) or `NDArray` (numpy)"
                )

    def _compute(self) -> dict:
        self._cat()
        output = self.compute(*self.arguments)
        self._convert_to_python_values(output)
        self.clear()

        return output

    def clear(self):
        self.arguments.clear()

    def add_batch(self, *args: Union[torch.Tensor, dict[Any, torch.Tensor]]):
        if len(self.arguments) == 0:
            # initialize lists
            self.arguments = [[] for _ in range(len(args))]

        for i, arg in enumerate(args):
            _type = type(arg)
            # transfer to CPU to avoid GPU memory issues
            if _type is torch.Tensor:
                if self.do_checks and len(self.arguments[i]) > 0:
                    prev = self.arguments[i][-1]
                    if prev.shape[1:] != arg.shape[1:]:
                        self.accelerator.end_training()
                        raise RuntimeError(
                            f"When appending metrics for main metric '{self.main_metric}', shape from "
                            f"previous tensor {tuple(prev.shape)} does not match current tensor {tuple(arg.shape)} "
                            "in second (or higher) dimension."
                        )
                arg = arg.cpu()
                if arg.is_floating_point() and self.cast is not None:
                    arg = arg.to(self.cast)

                self.arguments[i].append(arg)
            elif _type is dict:
                if self.do_checks and len(self.arguments[i]) > 0:
                    prev = self.arguments[i][-1]
                    for k, v in arg.items():
                        if prev[k].shape[1:] != v.shape[1:]:
                            self.accelerator.end_training()
                            raise RuntimeError(
                                f"When appending metrics for main metric '{self.main_metric}' in dataset '{k}', shape from "
                                f"previous tensor {tuple(prev[k].shape)} does not match current tensor {tuple(v.shape)} "
                                "in second (or higher) dimension."
                            )
                for k, v in arg.items():
                    v = v.cpu()
                    if v.is_floating_point() and self.cast is not None:
                        v = v.to(self.cast)

                    arg[k] = v  # ensure modification

                self.arguments[i].append(arg)
            else:
                raise NotImplementedError(f"'{_type}' type is not supported for metrics.")

    def _cat(self):
        for i, arg in enumerate(self.arguments):
            _type = type(arg[0])
            if _type is torch.Tensor:
                elem = torch.cat(arg)
            elif _type is dict:
                keys = set()
                for subarg in arg:
                    for k in subarg.keys():
                        keys.add(k)

                elem = defaultdict(list)
                for d in arg:
                    for k, v in d.items():
                        elem[k].append(v)

                elem = dict(elem)
                for k, v in elem.items():
                    elem[k] = torch.cat(v)
            else:
                raise NotImplementedError(f"'{_type}' type is not supported for metrics.")

            self.arguments[i] = elem


class MetricParallel(Metric):
    """Compute metrics in parallel."""

    def __init__(
        self, name: str, greater_is_better: bool = True, main_metric: Optional[str] = None, do_checks: bool = True
    ):
        """
        Set a module to compute metrics. All computations are done in parallel. When reporting values, these are averaged
        between all the processes.

        Args:
            name (`str`):
                Metric's module name.
            greater_is_better (`bool`, *optional*, defaults to `True`):
                Specify if the main metric is better when is greater.
            main_metric (`str`, *optional*, defaults to `None`):
                Determine which is the main metric key in your compute output. By default, main metric key will be
                equal to the 'name' parameter.
            do_checks (`bool`, *optional*, defaults to `True`):
                Enable shape checks when appending metrics. This can be disabled for small speed improvements.
        """
        super().__init__(name=name, greater_is_better=greater_is_better, main_metric=main_metric, do_checks=do_checks)
        self._parallel = True

    def _compute(self) -> dict:
        output = super()._compute()

        for k, v in output.items():
            # convert values to tensors in gpu for communication
            if isinstance(v, float):
                v = torch.tensor(
                    v, device=self.accelerator.device, dtype=torch.float64
                )  # fp64 to avoid dtype mismatch
            elif isinstance(v, np.ndarray):
                v = v.item()
                dtype = torch.float64 if isinstance(v, float) else torch.int64
                v = torch.tensor(v, device=self.accelerator.device, dtype=dtype)
            elif isinstance(v, torch.Tensor):
                # convert to correct dtype and move to gpu
                v = v.to(torch.float64) if v.is_floating_point() else v.to(torch.int64)
                v = v.to(self.accelerator.device)

            v = self.accelerator.reduce(v, reduction="mean")
            output[k] = v

        self._convert_to_python_values(output)
        return output
