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
import logging
from collections.abc import Mapping
from typing import Any, Callable, Optional, Union

import torch
from accelerate import DistributedType
from torch.utils.data import DataLoader, Dataset

from .dist_utils import Gatherer
from .metrics import Metric
from .model_wrapper import _DistributedDataParallel
from .modules import AcceleratorModule
from .utility import MASTER_PROCESS


class Evaluator:
    """
    Evaluator class for evaluating the model on a given dataset.

    Args:
        metrics (`Metric`, *optional*, defaults to `None`):
            The metrics to evaluate the model on. If not provided, the metrics will be the
            same as the ones used in evaluation during training.
        compile (`bool`, *optional*, defaults to `False`):
            Whether to compile the model.
        batch_size (`int`, *optional*, defaults to `1`):
            The batch size to use for evaluation.
        device_placement (`bool`, *optional*, defaults to `True`):
            Whether to place the batch on the device.
        num_workers (`int`, *optional*, defaults to `None`):
            The number of workers to use for evaluation in the dataloader.
        pin_memory (`bool`, *optional*, defaults to `True`):
            Whether to pin the memory of the batch.
        collate_fn (`Callable`, *optional*, defaults to `None`):
            The collate function to use for evaluation.
        prepare_batch (`bool`, *optional*, defaults to `True`):
            Whether to prepare the batch based on Mixed Precision. This only takes effect when using DeepSpeed.
        enable_prepare_logging (`bool`, *optional*, defaults to `False`):
            Whether to enable logging preparation (DeepSpeed).
    """

    def __init__(
        self,
        metrics: Optional[Union[Metric, list[Metric]]] = None,
        compile: bool = False,
        batch_size: int = 1,
        device_placement: bool = True,
        num_workers: Optional[int] = None,
        pin_memory: bool = True,
        collate_fn: Optional[Callable] = None,
        prepare_batch: bool = True,
        enable_prepare_logging: bool = False,
    ):
        self.metrics = None
        if metrics is not None:
            self.metrics = metrics if isinstance(metrics, list) else [metrics]
        self.compile = compile
        self.batch_size = batch_size
        self.device_placement = device_placement
        self.collate_fn = collate_fn
        from . import IS_GPU, accelerator

        self.num_workers = num_workers if num_workers is not None else accelerator.num_processes
        self.is_gpu = IS_GPU
        self.pin_memory = pin_memory if pin_memory and IS_GPU else False
        self.accelerator = accelerator
        self.prepare_batch = prepare_batch
        self.enable_prepare_logging = enable_prepare_logging
        self.gatherer = Gatherer()
        self._model_dtype = None

    def _prepare(self, module: AcceleratorModule, dataset: Dataset) -> tuple[AcceleratorModule, DataLoader]:
        is_deepspeed = self.accelerator.distributed_type == DistributedType.DEEPSPEED
        if not self.enable_prepare_logging and is_deepspeed:
            from deepspeed.utils import logger

            logger.setLevel(logging.WARNING)

        if is_deepspeed:
            # DeepSpeed requires contiguous parameters
            for param in module.model.parameters():
                if not param.is_contiguous():
                    param.data = param.data.contiguous()

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

        if not module._prepared:
            if torch.cuda.is_available():
                module.model.to(self.accelerator.device)
                module.model.eval()
                if module.teacher is not None:
                    module.teacher.eval()
                    module.teacher.to(self.accelerator.device)

            if self.compile:
                module.compile()

            module.model, module.teacher = self.accelerator.prepare(module.model, module.teacher)
            if self.accelerator.distributed_type == DistributedType.MULTI_GPU:
                module.model = _DistributedDataParallel(module.model)

            module._prepared = True

        dataloader = self.accelerator.prepare_data_loader(dataloader)

        if not self.device_placement:
            dataloader.device = torch.device("cpu")

        return module, dataloader

    def _add_metrics(self, metrics_dict: dict[str, Any], eval_logic_fn_name: str):
        if self.metrics is None:
            return

        for metric in self.metrics:
            if metric.main_metric not in metrics_dict:
                raise RuntimeError(f"Make sure to align '{eval_logic_fn_name}' with declared metrics.")
            metric_compute_arguments = metrics_dict[metric.main_metric]
            if not isinstance(metric_compute_arguments, tuple):
                metric_compute_arguments = (metric_compute_arguments,)

            if not metric._parallel:
                metric_compute_arguments = (
                    *(
                        (
                            self.gatherer.all_gather_dictionary(arg)
                            if isinstance(arg, dict)
                            else self.accelerator.gather_for_metrics(arg)
                        )
                        for arg in metric_compute_arguments
                    ),  # leave it as tuple
                )

                if MASTER_PROCESS and metric_compute_arguments[0] is not None:
                    metric.add_batch(*metric_compute_arguments)
            elif metric_compute_arguments[0] is not None:
                metric.add_batch(*metric_compute_arguments)

    def _compute_metrics(self, loss: torch.Tensor, dataloader: DataLoader, loss_implemented: bool) -> dict[str, Any]:
        num_batches = len(dataloader)
        loss /= num_batches
        results = {"loss": loss.item()} if loss_implemented else {}

        if self.metrics is not None:
            for metric in self.metrics:
                if not metric._parallel and MASTER_PROCESS:
                    # we don't want to call '_compute' for metrics that are implemented in main process,
                    # since the state on other processes is empty
                    metric_dict = metric._compute()
                    for m, v in metric_dict.items():
                        results[m] = v
                elif metric._parallel:
                    metric_dict = metric._compute()
                    # we are not fixing objects since in parallel mode they're already converted to python values
                    results.update(metric_dict)

        return results

    @torch.inference_mode()
    def evaluate(
        self,
        module: AcceleratorModule,
        dataset: Dataset,
        eval_logic_fn_name: str = "test_step",
        results_output: Optional[str] = "results.json",
    ):
        """
        Evaluates the model on the given dataset.

        Args:
            module (`AcceleratorModule`):
                The module to evaluate.
            dataset (`Dataset`):
                The dataset to evaluate on.
            eval_logic_fn_name (`str`, *optional*, defaults to `"test_step"`):
                The name of the method to use for evaluation.
            results_output (`str`, *optional*, defaults to `"results.json"`):
                The path to the file to save the results to. If `None`, the results will not be saved.

        Returns:
            `dict`:
                The results of the evaluation.
        """

        if not hasattr(module, eval_logic_fn_name):
            raise RuntimeError(f"Module {module} does not have a '{eval_logic_fn_name}' method.")

        dataset_length = len(dataset)
        self._model_dtype = next(module.model.parameters()).dtype
        module, dataloader = self._prepare(module, dataset)

        loss = torch.tensor(0, dtype=torch.float64, device=self.accelerator.device)
        _loss_implemented = False
        for batch in dataloader:
            if self.prepare_batch:
                batch = self._prepare_batch(batch)

            metrics_dict = module.test_step(batch)
            if isinstance(metrics_dict, torch.Tensor):
                # assume loss is the only metric
                loss += metrics_dict
            elif "loss" in metrics_dict:
                loss += metrics_dict.pop("loss")
                if not _loss_implemented:
                    _loss_implemented = True

            self._add_metrics(metrics_dict, eval_logic_fn_name)

        results = self._compute_metrics(loss, dataloader, _loss_implemented)

        if MASTER_PROCESS:
            print(f"\n\nEvaluation results on dataset with {dataset_length} samples:")
            for k, v in results.items():
                print(f"\t{k}: {v}")

        if results_output is not None:
            json.dump(results, open(results_output, "w"), indent=2, ensure_ascii=False)

        return results

    def _prepare_batch(self, batch: Any) -> Any:
        """
        Prepare elements in a batch based on Mixed Precision. This function only takes effect when using DeepSpeed.
        """
        if self.accelerator.distributed_type != DistributedType.DEEPSPEED:
            return batch

        return self._prepare_nested_batch(batch)

    def _prepare_nested_batch(self, batch: Any) -> Any:
        """
        Prepare nested batch. This function is derived from `transformers` library
        (https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py).
        """
        if isinstance(batch, Mapping):
            return type(batch)({k: self._prepare_nested_batch(v) for k, v in batch.items()})
        elif isinstance(batch, (tuple, list)):
            return type(batch)(self._prepare_nested_batch(v) for v in batch)
        elif isinstance(batch, torch.Tensor):
            kwargs = {"device": self.accelerator.device}
            if torch.is_floating_point(batch) or torch.is_complex(batch):
                kwargs.update({"dtype": self._model_dtype})

            return batch.to(**kwargs)
        return batch
