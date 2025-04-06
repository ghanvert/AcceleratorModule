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

import inspect
import math
import os
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from accelerate import DistributedType
from accelerate.utils import LoggerType, ProjectConfiguration
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset

from .callbacks import Callback, CallbackMaster
from .dist_utils import Gatherer, rprint
from .hyperparameters import HyperParameters
from .metrics import Metric
from .model_wrapper import _DistributedDataParallel
from .modules import AcceleratorModule
from .monitor import Monitor
from .states import LossState, TrainingState
from .tqdm import tqdm
from .utility import DEBUG_MODE, MASTER_PROCESS
from .utils import (
    cleanup,
    combine_dicts,
    filter_kwargs,
    get_number_and_unit,
    get_seed,
    is_url,
    operator_map,
    set_seed,
    time_prefix,
)


CHECKPOINT_DIR = "checkpoint"
STATE_FILE = "state.json"
TRAIN_LOSS_STATE_FILE = "train_loss_state.pt"
_bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} - ETA: {remaining}{postfix} - {rate_s}"
_tqdm_kwargs = {"leave": False, "ncols": 100, "bar_format": _bar_format}


class Trainer:
    """Class to implement full training process."""

    def __init__(
        self,
        hps_config: Union[str, dict, HyperParameters],
        model_path: str,
        track_name: Optional[str] = None,
        enable_checkpointing: bool = True,
        resume: Optional[bool] = None,
        disable_model_saving: bool = False,
        patience: Optional[int] = None,
        evaluate_every_n_steps: Optional[int] = None,
        checkpoint_every: Optional[str] = "epoch",
        logging_dir: str = "logs",
        log_with: Optional[Union[Any, list]] = None,
        log_every: Optional[int] = 1,
        grad_accumulation_steps: Optional[int] = None,
        clip_grad: Optional[float] = None,
        set_to_none: bool = True,
        shuffle_train: bool = True,
        sampler: Optional[Union[Any, list]] = None,
        collate_fn_train: Optional[Callable] = None,
        collate_fn_val: Optional[Callable] = None,
        max_shard_size: str = "10GB",
        safe_serialization: bool = False,
        compile: bool = False,
        compile_kwargs: Optional[dict[str, Any]] = None,
        safe_mode: bool = True,
        train_loss_metric_name: str = "train_loss",
        val_loss_metric_name: str = "val_loss",
        dataloader_pin_memory: bool = True,
        dataloader_num_workers: Optional[int] = None,
        dataloader_drop_last: bool = False,
        eval_when_finish: bool = True,
        eval_when_start: bool = False,
        monitor: Optional[Monitor] = None,
        metrics: Optional[Union[Metric, list[Metric], dict[Any, Union[Metric, list[Metric]]]]] = None,
        cleanup_cache_every_n_steps: Optional[int] = None,
        callback: Optional[Union[Callback, list[Callback]]] = None,
        additional_tracker_config: Optional[dict[str, Any]] = None,
        batch_device_placement: bool = True,
        **kwargs: Optional[Any],
    ):
        """
        Trainer constructor to set configuration.

        Args:
            hps_config (`str`, `dict`, or `HyperParameters`):
                YAML hyperparameters file path, dictionary or `HyperParameters`.
            model_path (`str`):
                Path to save model.
            track_name (`str`, *optional*, defaults to `None`):
                Track name for trackers. If set to `None` (default), the track name will be
                the model's folder name.
            enable_checkpointing (`bool`, *optional*, defaults to `True`):
                Enable checkpointing.
            resume (`bool`, *optional*, defaults to `None`):
                Whether to resume from checkpoint. Default option is `None`, which means resuming from checkpoint
                will be handled automatically, whether the checkpoint directory exists or not.
            disable_model_saving (`bool`, *optional*, defaults to `False`):
                Disable any model saving registered (by default, `"best_valid_loss"` is registered, or if there are none evaluations to do,
                default will be `"best_train_loss"`).
            patience (`int`, *optional*, defaults to `None`):asdas
                Set up a patience parameter for model savings. If set, every model saving will check if the previous metric was higher.
                If the metric has not improved over the N model savings (`patience`), then the training process will stop.
            evaluate_every_n_steps (`int`, *optional*, defaults to `None`):
                Evaluate model in validation dataset (if implemented) every N steps. If this is set
                to `None` (default option), evaluation will happen at the end of every epoch.
            checkpoint_every (`str`, *optional*, defaults to `epoch`):
                Checkpoint every N epochs, steps or evaluations. Requires a number and a unit in a string.
                The following examples are valid:

                - `"epoch"`, `"ep"`, `"1epoch"`, `"1ep"`, `"1 epoch"`, `"1 ep"`: 1 Epoch
                - `"step"`, `"st"`, `"1step"`, `"1st"`, `"1 step"`, `"1 st"`: 1 Step
                - `"evaluation"`, `"eval"`, `"1evaluation"`, `"1eval"`, `"1 evaluation"`, `"1 eval"`: 1 Evaluation

                (a character `s` at the end of the string is also valid)

                If set to `None`, checkpointing will be disabled.
            logging_dir (`str`, *optional*, defaults to `logs`):
                Path where to save logs to show progress. It can be an IP address (local or remote), HTTP or HTTPS link,
                or simply a directory.
            log_with (`accmt.tracker` or `list`, *optional*, defaults to `None`):
                Logger to log metrics. It can be one of the following imports from accmt:
                    - `MLFlow`

                NOTE: MLFlow is the only one supported right now. Other trackers are not currently available.
            log_every (`int`, *optional*, defaults to `1`):
                Log train loss every N steps. If set to `-1`, training loss will be logged at the end of every epoch.
            grad_accumulation_steps (`int`, *optional*, defaults to `None`):
                Accumulate gradients for N steps. Useful for training large models and simulate
                large batches when memory is not enough. If set to `None` or `1`, no accumulation will be perfomed.
            clip_grad (`float`, *optional*, defaults to `None`):
                Performs gradient clipping in between backpropagation and optimizer's step function. This feature is disabled when
                using DeepSpeed, because it handles gradient clipping in the configuration file. If you wan't to configure gradient
                clipping, you might want to use Accelerate's CLI to create a new config file.
            set_to_none (`bool`, *optional*, defaults to `True`):
                From PyTorch documentation: "instead of setting to zero, set the grads to None. This will
                in general have lower memory footprint, and can modestly improve performance." Some
                optimizers have a different behaviour if the gradient is 0 or None. See PyTorch docs
                for more information: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
            shuffle_train (`bool`, *optional*, defaults to `True`):
                Whether to shuffle train DataLoader.
            sampler (`list` or `Any`, *optional*, defaults to `None`):
                Sampler (or list of samplers) for train DataLoader.
            collate_fn_train (`Callable`, *optional*, defaults to `None`):
                Collate function to be implemented in train dataloader.
            collate_fn_val (`Callable`, *optional*, defaults to `None`):
                Collate function to be implemented in validation dataloader.
            max_shard_size (`str`, *optional*, defaults to `10GB`):
                Max model shard size to be used.
            safe_serialization (`bool`, *optional*, defaults to `False`):
                Whether to save model using safe tensors or the traditional PyTorch way. If `True`, some tensors
                will be lost.
            compile (`bool`, *optional*, defaults to `False`):
                Whether to call `torch.compile` on model (and teacher, if implemented).
            compile_kwargs (`dict`, *optional*, defaults to `None`):
                `torch.compile` kwargs for additional customization.
            safe_mode (`bool`, *optional*, defaults to `True`):
                Run forward passes of the model in safe mode. This means that the forward pass of the model will run
                through the corresponding wrapper (DDP, FSDP or DeepSpeedEngine). If not running in safe mode, forward pass
                will skip the wrapper and run directly on the module (instance of `nn.Module`). Running with safe mode disabled
                will slightly improve throughput, although gradients consistency and mixed precision could be affected because
                skipping the wrapper's forward pass might skip internal parallel functionality.

                **NOTE**: This parameter takes no effect running with FSDP since forward passes are already done through this wrapper.
            train_loss_metric_name (`str`, *optional*, defaults to `train_loss`):
                Metric name for train loss in logs.
            val_loss_metric_name (`str`, *optional*, defaults to `val_loss`):
                Metric name for validation loss in logs.
            dataloader_pin_memory (`bool`, *optional*, defaults to `True`):
                Enables pin memory option in DataLoader (only if GPU is enabled).
            dataloader_num_workers (`int`, *optional*, defaults to `None`):
                Number of processes for DataLoader. This defaults to `None`, meaning the number of workers will be equal to the
                number of processes set for training.
            dataloader_drop_last (`bool`, *optional*, defaults to `False`):
                Whether to drop last batch on DataLoader or not.
            eval_when_finish (`bool`, *optional*, defaults to `True`):
                At the end of training, evaluate model on validation dataset (if available). This option is only valid when
                `evaluate_every_n_steps` is not `None`.
            eval_when_start (`bool`, *optional*, defaults to `False`):
                Start training with evaluation (if available).
            monitor (`Monitor` or `dict`, *optional*, defaults to `None`):
                Monitor arguments to keep track of variables during training. If not specified, 'train_loss' and 'validation_loss' will
                be set to `True` by default.

                NOTE: Learning rate, GPU and CPU monitoring will only be reported during training, not evaluation. Also, GPU and CPU
                monitoring will only be reported on main process (index 0).
            metrics (`Metric`, `list` or `dict`, *optional*, defaults to `None`):
                List of additional metrics of type 'Metric' to track. When doing multiple evaluations, this should be a dictionary
                of metrics (or list of metrics), where each key corresponds to the dataset to evaluate (specified in `val_dataset`
                in `fit` function) and the value corresponds to a `Metric` or list of metrics. If metrics are given as only `Metric`
                or list of metrics, these metrics will apply for all evaluations. If you want specific metrics for specific evaluations,
                consider dividing your metrics per validation dataset in a dictionary.
            cleanup_cache_every_n_steps (`int`, *optional*, defaults to `None`):
                Cleanup CPU and CUDA caches every N steps. Default is no cleanup.

                NOTE: On every epoch and evaluation call we cleanup cache.
            callback (`Callback` or `list`, *optional*, defaults to `None`):
                `Callback` or callbacks to implement.
            additional_tracker_config (`dict`, *optional*, defaults to `None`):
                Additional configuration specification for tracker (e.g. hyper-parameters).
            batch_device_placement (`bool`, *optional*, defaults to `True`):
                Move batches to correct device automatically. If `False`, batches will be in CPU.
            kwargs (`Any`, *optional*):
                Extra arguments for specific `init` function in Tracker, e.g. `run_name`, `tags`, etc.
        """
        # do some previous checks
        self.log_with = log_with if isinstance(log_with, list) else [log_with]
        self.log_with = [tracker for tracker in self.log_with if tracker is not None]

        # TODO we have to add support for other trackers (and multiple).
        assert len(self.log_with) < 2, "For now, we only support one tracker in 'log_with'."
        if len(self.log_with) == 1:
            assert self.log_with[0].tracker == LoggerType.MLFLOW, "Only MLFlow is supported as tracker."

        assert isinstance(hps_config, (str, dict, HyperParameters)), (
            "'hps_config' needs to be either a string, dictionary or HyperParameters class."
        )
        assert clip_grad is None or isinstance(clip_grad, float), "'clip_grad' argument needs to be a float."

        from . import IS_GPU, accelerator

        self.is_gpu = IS_GPU
        self.accelerator = accelerator

        self.hps = HyperParameters.from_config(hps_config) if isinstance(hps_config, (str, dict)) else hps_config
        self.track_name = track_name
        self.checkpoint_path = os.path.join(model_path, CHECKPOINT_DIR)
        self.model_path = model_path
        self.resume = (
            (
                resume
                if resume is not None
                else os.path.exists(self.checkpoint_path) and len(os.listdir(self.checkpoint_path)) > 0
            )
            if DEBUG_MODE < 3
            else False
        )

        self.metrics: dict[Any, list[Metric]] = metrics
        self.disable_model_saving = disable_model_saving
        self.model_saving: dict[
            str, tuple[float, float]
        ] = {}  # key: model saving, value: (saving_below, saving_above)

        self.patience = patience if patience is not None else -1
        if self.patience == 0:
            self.accelerator.end_training()
            raise ValueError("The 'patience' argument in Trainer should have a value greater than 0.")

        self.evaluate_every_n_steps = evaluate_every_n_steps
        self.enable_checkpointing = enable_checkpointing if DEBUG_MODE < 3 else False
        self.checkpoint_every, self.checkpoint_strat = get_number_and_unit(checkpoint_every)
        self.logging_dir = logging_dir
        self.log_every = log_every
        self.grad_accumulation_steps = grad_accumulation_steps if grad_accumulation_steps is not None else 1
        self.accelerator.gradient_accumulation_steps = self.grad_accumulation_steps
        assert clip_grad is None or isinstance(clip_grad, float), "'clip_grad' argument needs to be a float."
        self.clip_grad = clip_grad if clip_grad is not None else 0.0
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            self.accelerator.deepspeed_plugin.deepspeed_config["gradient_clipping"] = self.clip_grad
        self.set_to_none = set_to_none
        self.shuffle_train = shuffle_train
        self.sampler = sampler
        self.collate_fn_train = collate_fn_train
        self.collate_fn_val = collate_fn_val
        self.max_shard_size = max_shard_size
        self.safe_serialization = safe_serialization
        self.compile = compile
        self.compile_kwargs = compile_kwargs if compile_kwargs is not None else {}
        self.safe_mode = safe_mode
        self.train_loss_metric_name = train_loss_metric_name
        self.val_loss_metric_name = val_loss_metric_name
        self.dataloader_pin_memory = dataloader_pin_memory if IS_GPU else False
        self.dataloader_num_workers = (
            dataloader_num_workers if dataloader_num_workers is not None else self.accelerator.num_processes
        )
        if DEBUG_MODE > 0 and self.dataloader_num_workers != 0:
            self.dataloader_num_workers = (
                0  # force when debugging to not have problems with dataloader during breakpoints
            )
        self.dataloader_drop_last = dataloader_drop_last
        self.samplers = sampler
        self.eval_when_finish = eval_when_finish
        self.eval_when_start = eval_when_start if DEBUG_MODE < 4 else False
        self.monitor = monitor if isinstance(monitor, Monitor) else Monitor.from_config(monitor)
        self.monitor.grad_norm = (
            self.monitor.grad_norm if self.accelerator.distributed_type != DistributedType.DEEPSPEED else False
        )
        if self.monitor.grad_norm and self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            rprint(
                "[WARNING] Gradient norm monitoring is not yet supported when running with DeepSpeed. Setting it to False."
            )
            self.monitor.grad_norm = False
        self.cleanup_cache_every_n_steps = cleanup_cache_every_n_steps
        callback = callback if callback is not None else Callback()
        callback = callback if isinstance(callback, list) else [callback]
        self.callback = CallbackMaster(callback)
        self.additional_tracker_config = additional_tracker_config if additional_tracker_config is not None else {}
        self.batch_device_placement = batch_device_placement
        self.init_kwargs = kwargs

        self.accelerator.project_configuration = ProjectConfiguration(
            project_dir=".", logging_dir=logging_dir, total_limit=1
        )

        self._logging = len(self.log_with) > 0
        if self._logging and DEBUG_MODE < 1:
            self._init_trackers()

        self.state = TrainingState()
        self.gatherer = Gatherer()
        # adding a total (at maximum) of 64 bytes for additional tensors
        self.train_loss_state = LossState(self.accelerator, self.accelerator.device, self.log_every, pin_memory=IS_GPU)
        self.val_loss_state: dict[Any, LossState] = None  # prepare val loss states in 'fit' function

        self._checkpointing_every_n_steps = self.enable_checkpointing and self.checkpoint_strat == "step"
        self._checkpointing_after_evaluation = self.enable_checkpointing and self.checkpoint_strat == "eval"
        self._checkpointing_when_epoch_ends = self.enable_checkpointing and self.checkpoint_strat == "epoch"

        self._scheduler: LRScheduler = None
        self._optimizer: Optimizer = None

        self._multiple_evaluations = False

    def fit(
        self,
        module: Union[AcceleratorModule, str, Union[tuple[str, str], tuple[str, Any]]],
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Union[Dataset, list[Dataset], dict[str, Dataset]]] = None,
        **kwargs: Any,
    ):
        """
        Function to train a given `AcceleratorModule`.

        Args:
            module (`AcceleratorModule`, `str` or `tuple`):
                `AcceleratorModule` class containig the training logic. This can also be a string specifying a
                HuggingFace model, or a tuple of type (model, type), where 'model' is a string for the HuggingFace model,
                and 'type' is a string or class (from transformers library) for the model type.
            train_dataset (`torch.utils.data.Dataset`, *optional*, defaults to `None`):
                `Dataset` class from PyTorch containing the train dataset logic. If not provided, then
                `get_train_dataloader` from `module` will be used to get the train DataLoader.
            val_dataset (`torch.utils.data.Dataset`, `list` or `dict`, *optional*, defaults to `None`):
                `Dataset` class from PyTorch containing the validation dataset logic. This can also be a list or a dictionary
                of `Dataset`, in that case, multiple evaluations will run following the logic of `validation_step` and
                specified metrics. Metric names reported for a multiple evaluation setting will add a '_' followed by a key
                related to the dataset (e.g. 'accuracy_1' or 'accuracy_another_dataset').

                If this dataset is not specified, then the validation logic of `AcceleratorModule`
                (if specified) will be skipped.
            kwargs (`Any`):
                Keyword arguments for `from_pretrained` function for model initialization.
        """
        # reset loss states in case of another fit function call in the script
        self.train_loss_state.reset()
        if self.val_loss_state is not None:
            for v in self.val_loss_state.values():
                v.reset()

        module = self._get_module(module, **kwargs)

        model = module.model
        if model is None or not isinstance(model, nn.Module):
            self.accelerator.end_training()
            raise RuntimeError(
                "`AcceleratorModule` subclass requires `self.model` and needs to be an instance of `nn.Module`."
            )

        teacher = module.teacher
        if torch.cuda.is_available():
            model.to(self.accelerator.device)
            if teacher is not None:
                teacher.eval()
                teacher.to(self.accelerator.device)

        module.state = self.state
        module.accelerator = self.accelerator
        module.device = self.accelerator.device

        if MASTER_PROCESS and DEBUG_MODE < 3:
            os.makedirs(self.model_path, exist_ok=True)

        val_dataset = val_dataset if val_dataset is None or isinstance(val_dataset, (list, dict)) else [val_dataset]
        self._multiple_evaluations = val_dataset is not None and len(val_dataset) > 1
        train_dataloader, val_dataloader = self._get_dataloaders(module, train_dataset, val_dataset)

        self.metrics = self._prepare_metrics(self.metrics, val_dataloader)
        if len(self.model_saving) == 0:
            if val_dataloader is not None:
                self.model_saving["best_valid_loss"] = (float("inf"), float("-inf"))
            else:
                self.model_saving["best_train_loss"] = (float("inf"), float("-inf"))
        self.state.patience_left = {k: self.patience for k in self.model_saving.keys()}
        if self.metrics is not None:
            for k, v in self.metrics.items():
                self.state.additional_metrics[k] = {m.main_metric: 0 for m in v}
        else:
            for k in val_dataloader.keys():
                self.state.additional_metrics[k] = {}

        if self.resume:
            training_state_path = os.path.join(self.checkpoint_path, STATE_FILE)
            loss_tracker_path = os.path.join(self.checkpoint_path, TRAIN_LOSS_STATE_FILE)
            self.state.load(training_state_path)
            self.train_loss_state.load(loss_tracker_path)

        if self.state.finished:
            self.accelerator.end_training()
            raise RuntimeError("Training process has been flagged as finished.")

        module.state = self.state

        self.monitor._set_extra(self.accelerator, self.state, self.train_loss_metric_name, self.val_loss_metric_name)

        if self.accelerator.distributed_type == DistributedType.FSDP:
            # preparing model before dataloaders is only supported by FSDP apparently, and this is the
            # recommended setting to prepare training.
            model = self.accelerator.prepare_model(model)

        self.val_loss_state = {
            k: LossState(
                self.accelerator, self.accelerator.device, -1, include_per_batch=False, pin_memory=self.is_gpu
            )
            for k in val_dataloader.keys()
        }

        optimizer = self._get_optimizer(module)
        scheduler = self._get_scheduler(
            module, optimizer, round(len(train_dataloader) / self.accelerator.num_processes), self.hps.epochs
        )

        model, teacher, train_dataloader, val_dataloader, optimizer, scheduler = self._prepare(
            module,
            model,
            teacher,
            train_dataloader,
            val_dataloader,
            optimizer,
            scheduler,
            batch_device_placement=self.batch_device_placement,
        )

        self._scheduler = scheduler
        self._optimizer = optimizer
        module.scheduler = scheduler
        module.optimizer = optimizer

        if self.log_every < 0:  # report training loss at the last step (or end of an epoch)
            self.log_every = len(train_dataloader)

        for callback in self.callback.children:
            callback.module = module
            callback.trainer = self
            callback.state = self.state

        self.callback.on_fit_start()
        self.loop(module, model, train_dataloader, val_dataloader, optimizer, scheduler)
        self.state.finished = True
        self.callback.on_fit_end()

    def loop(
        self,
        module: AcceleratorModule,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[dict[Any, DataLoader]],
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler],
    ):
        """Runs a training loop."""
        if self.state.evaluations_done == 0 and self.eval_when_start:
            self.eval(module, model, val_dataloader)

        for _ in self.epoch_iterator():
            for batch in self.batch_iterator(train_dataloader, model):
                self._train_logic(module, model, optimizer, batch, scheduler)

                if (
                    self.evaluate_every_n_steps is not None
                    and (self.state.global_step + 1) % self.evaluate_every_n_steps == 0
                ):
                    self.eval(module, model, val_dataloader)

            if self.evaluate_every_n_steps is None or (self.eval_when_finish and self.state.is_last_epoch):
                self.eval(module, model, val_dataloader)

    @torch.inference_mode()
    def eval(self, module: AcceleratorModule, model: nn.Module, dataloader: Optional[dict[Any, DataLoader]]):
        """Runs evaluation on a given dataloader."""
        if DEBUG_MODE >= 5 or self.state.finished or dataloader is None:
            return

        if model.training:
            model.eval()

        cleanup()
        self.callback.on_evaluation_start()
        for k, val_dataloader in dataloader.items():
            val_str = f" ({k}) " if self._multiple_evaluations else " "
            for i, batch in tqdm(
                iterable=enumerate(val_dataloader),
                total=len(val_dataloader),
                desc=f"📊{val_str}Evaluating in Epoch {self.state.epoch + 1}/{self.hps.epochs}",
                position=1,
                colour="cyan",
                **_tqdm_kwargs,
            ):
                self.state.val_step = i
                self.state.is_last_validation_batch = i == len(val_dataloader) - 1
                self._validation_logic(module, k, batch)

            self.state.additional_metrics[k]["valid_loss"] = self.val_loss_state[k].get_total_loss()

            # only master process will be in charge of calculating metrics to avoid system overhead
            if MASTER_PROCESS and self.metrics is not None:
                for metric in self.metrics[k]:
                    metric_dict = metric._compute()

                    for m, v in metric_dict.items():
                        if not isinstance(v, (float, int, torch.Tensor, np.ndarray)):
                            self.accelerator.end_training()
                            raise ValueError(
                                f"Value in metric's dict does not accept {type(v)}, only "
                                f"`float`, `int`, `torch.Tensor` (torch) or `NDArray` (numpy)"
                            )

                        self.state.additional_metrics[k][m] = (
                            v if not isinstance(v, (torch.Tensor, np.ndarray)) else v.item()
                        )

            # re-format metrics, instead of a dict dataset_key (key) and metrics (dictionary value), gather
            # all metrics into a single dictionary with the format {metric__dataset_key: value}.
            # e.g. {"accuracy__dataset1": 0.21, "accuracy__dataset2": 0.67}
            log_dict = {}
            for _metric_name, _value in self.state.additional_metrics[k].items():
                if _metric_name.startswith("best_"):
                    continue
                _metric_name = f"{_metric_name}__{k}" if self._multiple_evaluations else _metric_name
                log_dict[_metric_name] = _value

            self.monitor.log_additional_metrics(log_dict)

        self.state.evaluations_done += 1

        should_save_model = not (
            self.eval_when_start and self.state.evaluations_done == 1
        )  # not doing first requested evaluation

        # save model
        if self.model_saving is not None and should_save_model and DEBUG_MODE < 3:
            self._save_model_on_criteria(model)
        else:
            # reset total loss state for validation since it's not being used
            for k in self.val_loss_state.keys():
                self.val_loss_state[k].total_loss.zero_()
                self.val_loss_state[k].num_steps.zero_()

        self.state.val_step = 0

        # flag as finished if doing very last evaluation
        self.state.finished = self.state.is_last_training_batch and self.state.is_last_epoch

        if self._checkpointing_after_evaluation and should_save_model:
            self._save_checkpoint(
                self.state.epoch + (0 if not self.state.is_end_of_epoch else 1),
                self.state.train_step + (1 if not self.state.is_end_of_epoch else 0),
                self.state.global_step + (1 if not self.state.is_end_of_epoch else 0),
                self.state.evaluations_done,
                finished=self.state.finished,
            )

        self.callback.on_evaluation_end()

    def _save_model_on_criteria(self, model: nn.Module):
        """Save model depending on criteria defined in `model_saving`"""
        self.accelerator.wait_for_everyone()

        train_loss = self.train_loss_state.get_total_loss()
        can_save = not (self.eval_when_start and self.state.evaluations_done == 1)

        def _check_and_save(model_saving: str):
            _model_saving = model_saving
            model_saving_without_prefix = model_saving.removeprefix("best_")
            # we already have all metrics calculated per dataset in self.state.additional_metrics
            metrics_and_datasets = defaultdict(
                list
            )  # e.g. {"accuracy": ["dataset1", "dataset2"], "metric": [dataset_keys, ...]}
            for metric in model_saving_without_prefix.split("/"):
                metric, *datasets = metric.split("@")
                if len(datasets) == 0:
                    # if datasets are not specified for a metric, then it means that we need to average
                    # across all datasets
                    datasets = self.state.additional_metrics.keys()

                for dataset in datasets:
                    metrics_and_datasets[metric].append(dataset)

            # now create a buffer per metric, where each value in the buffer corresponds to the
            # metric found in a dataset
            metric_buffer = defaultdict(list)  # e.g. {"accuracy": [0.2, 0.5], "metric": [values, ...]}
            for dataset_key, metrics_dict in self.state.additional_metrics.items():
                for metric, value in metrics_dict.items():
                    if dataset_key in set(metrics_and_datasets[metric]):
                        metric_buffer[metric].append(value)

            # now average those metrics in buffer
            metric_avgs = {k: (np.mean(v) if len(v) > 1 else v[0]) for k, v in metric_buffer.items()}

            _metrics = [ms.split("@")[0] for ms in model_saving_without_prefix.split("/")]
            count = 0
            for metric in _metrics:
                best_metric_str = f"best_{metric}"
                comparator = self._get_comparator(metric) if metric != "valid_loss" else "<"
                compare = operator_map[comparator]
                new = metric_avgs[metric]
                # calculate average between previous metrics in wanted datasets
                prev = []
                for dataset_key in set(metrics_and_datasets[metric]):
                    if best_metric_str not in self.state.additional_metrics:
                        # only register best metrics in wanted datasets
                        self.state.additional_metrics[dataset_key][best_metric_str] = (
                            float("inf") if comparator in {"<", "<=", "=="} else float("-inf")
                        )

                    prev.append(self.state.additional_metrics[dataset_key][best_metric_str])

                prev = np.mean(prev) if len(prev) > 1 else prev[0]
                saving_below, saving_above = self.model_saving[_model_saving]
                is_better = compare(new, prev) and new < saving_below and new > saving_above
                if is_better:
                    count += 1

                # register best metrics for all wanted datasets
                for dataset, new_metric_calculated in zip(metrics_and_datasets[metric], metric_buffer[metric]):
                    local_prev = self.state.additional_metrics[dataset][best_metric_str]
                    if compare(new_metric_calculated, local_prev):
                        self.state.additional_metrics[dataset][best_metric_str] = new_metric_calculated

                if count == len(_metrics):
                    # all these metrics have improved
                    if MASTER_PROCESS and can_save and not self.disable_model_saving:
                        model_path = os.path.join(self.model_path, _model_saving.replace("/", "__"))
                        self._save_model(model, model_path)
                elif can_save and self.state.patience_left[_model_saving] > 0:
                    self.state.patience_left[_model_saving] -= 1

        if len(self.model_saving) > 0:
            if train_loss < self.state.best_train_loss:
                self.state.best_train_loss = train_loss
                if "best_train_loss" in self.model_saving:
                    if MASTER_PROCESS and can_save and not self.disable_model_saving:
                        model_path = os.path.join(self.model_path, "best_train_loss")
                        self._save_model(model, model_path)
            elif (
                can_save and "best_train_loss" in self.model_saving and self.state.patience_left["best_train_loss"] > 0
            ):
                self.state.patience_left["best_train_loss"] -= 1

            for model_saving in self.model_saving:  # TODO we could implement a tqdm bar maybe...
                _check_and_save(model_saving)

            # if all model savings have no patience anymore, finish training process
            count = 0
            for model_saving in self.model_saving.keys():
                count += self.state.patience_left[model_saving] == 0

            self.accelerator.wait_for_everyone()
            if count == len(self.model_saving):
                rprint("Ran out of patience. Process finished.")
                self.state.finished = True
                if MASTER_PROCESS:
                    state_in_checkpoint = os.path.join(self.checkpoint_path, STATE_FILE)
                    self.state.save(state_in_checkpoint)
                    for model_saving in self.model_saving:
                        model_saving_path = os.path.join(self.model_path, model_saving)
                        os.makedirs(model_saving_path, exist_ok=True)
                        model_saving_path = os.path.join(model_saving_path, STATE_FILE)
                        self.state.save(model_saving_path)
                self.accelerator.end_training()
                exit(0)

    def _save_model(self, model: nn.Module, path: str):
        """Save model inside a path."""
        tqdm.write(f"\r{time_prefix()} Saving model...")
        os.makedirs(path, exist_ok=True)

        unwrapped_model = self.accelerator.unwrap_model(model)
        state_dict = unwrapped_model.state_dict() if not self.compile else unwrapped_model._orig_mod.state_dict()
        if hasattr(unwrapped_model, "save_pretrained"):  # special function for models from transformers library
            unwrapped_model.save_pretrained(
                path,
                is_main_process=True,
                state_dict=state_dict,
                max_shard_size=self.max_shard_size,
                save_function=self.accelerator.save,
                safe_serialization=self.safe_serialization,
            )
        else:
            pt_state_dict = os.path.join(path, "pytorch_model.pt")
            self.accelerator.save(state_dict, pt_state_dict, safe_serialization=self.safe_serialization)

        training_state_path = os.path.join(path, STATE_FILE)
        self.state.save(training_state_path)

        tqdm.write(f"\033[A\033[K{time_prefix()} Model saved.")

    def _validation_logic(self, module: AcceleratorModule, dataloader_key: Any, batch: Any):
        """Runs all the validation logic."""
        self.callback.on_before_validation_step(batch)
        metrics = module.validation_step(dataloader_key, batch)
        if isinstance(metrics, torch.Tensor):
            # assume it's loss value, so convert wrap it into a dictionary
            metrics = {"loss": metrics}
        self.callback.on_after_validation_step()
        # track loss
        loss = metrics["loss"].detach()
        self.val_loss_state[dataloader_key].add_total_loss(loss)

        # track metrics
        if self.metrics is not None:
            for metric in self.metrics[dataloader_key]:
                if metric.main_metric not in metrics:
                    self.accelerator.end_training()
                    raise RuntimeError("Make sure to align 'validation_step' with declared metrics.")
                metric_compute_arguments = metrics[metric.main_metric]
                if not isinstance(metric_compute_arguments, tuple):
                    metric_compute_arguments = (metric_compute_arguments,)

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

    def _train_logic(
        self,
        module: AcceleratorModule,
        model: nn.Module,
        optimizer: Optimizer,
        batch: Any,
        scheduler: Optional[LRScheduler],
    ):
        """Runs all the training logic."""
        self.callback.on_before_training_step(batch)
        with self.accelerator.accumulate(model) if self.grad_accumulation_steps > 1 else nullcontext():
            # forward pass
            loss = module.training_step(batch)
            self.callback.on_after_training_step()

            # track
            _loss = loss.detach()
            self.train_loss_state.add_batch_loss(_loss)
            self.train_loss_state.add_total_loss(_loss)

            self.callback.on_before_backward(loss)
            if not module._extended:
                # backpropagation
                self.accelerator.backward(loss)
                self.callback.on_after_backward()

            if self.accelerator.sync_gradients and self.clip_grad > 0.0:
                self.accelerator.clip_grad_norm_(model.parameters(), self.clip_grad)

            if self.state.global_step % self.log_every == 0:
                batch_loss = self.train_loss_state.get_batch_loss()

                norm = None
                if MASTER_PROCESS and self.monitor.grad_norm:
                    norm = self._get_grad_norm(model)

                self.monitor.log_train_loss_and_grad_norm(batch_loss, norm)

            if not module._extended:
                self.callback.on_before_optimizer_step(optimizer)
                optimizer.step()
                self.callback.on_after_optimizer_step(optimizer)
                if scheduler is not None:
                    self.callback.on_before_scheduler_step(scheduler)
                    scheduler.step()
                    self.callback.on_after_scheduler_step(scheduler)

                # reset gradients
                self.callback.on_before_optimizer_step(optimizer)
                optimizer.zero_grad(set_to_none=self.set_to_none)
                self.callback.on_after_zero_grad(optimizer)

    def batch_iterator(self, dataloader: DataLoader, model: nn.Module):
        """Batch iterator for training handling checkpointing."""
        if not model.training:
            model.train()

        if self.shuffle_train:
            global_seed = get_seed(default=0)
            set_seed(global_seed + self.state.epoch)
            dataloader.set_epoch(self.state.epoch)

        _dataloader = self.accelerator.skip_first_batches(dataloader, self.state.train_step)

        cleanup()
        start = self.state.train_step
        if len(_dataloader) > 0:
            for i, batch in tqdm(
                iterable=enumerate(_dataloader, start),
                total=len(dataloader),
                initial=start,
                desc=f"🚀 Training in Epoch {self.state.epoch + 1}/{self.hps.epochs}",
                position=0,
                colour="green",
                **_tqdm_kwargs,
            ):
                self.state.train_step = i
                self.state.is_last_training_batch = i == len(dataloader) - 1

                if self.state.global_step % self.log_every:
                    lr = (
                        self._scheduler.get_last_lr()[-1]
                        if self._scheduler is not None
                        else self._optimizer.param_groups[0]["lr"]
                    )

                    # TODO we can fuse these functions to only report once to the server
                    self.monitor.log_learning_rate(lr)
                    self.monitor.log_cpu_utilization()
                    self.monitor.log_gpu_utilization()

                yield batch

                if (
                    self.cleanup_cache_every_n_steps is not None
                    and (self.state.global_step + 1) % self.cleanup_cache_every_n_steps == 0
                ):
                    cleanup()

                if self._checkpointing_every_n_steps and (self.state.global_step + 1) % self.checkpoint_every == 0:
                    self._save_checkpoint(
                        self.state.epoch,
                        self.state.train_step + 1,
                        self.state.global_step + 1,
                        self.state.evaluations_done,
                    )

                self.state.global_step += 1

        # if length of _dataloader is 0, then we do not iterate

        self.state.is_end_of_epoch = True
        self.state.train_step = 0

    def _save_checkpoint(
        self, epoch: int, train_step: int, global_step: int, evaluations_done: int, finished: bool = False
    ):
        """Save checkpoint at a given point in time (`epoch` and `train_step`)."""
        self.callback.on_save_checkpoint()
        if MASTER_PROCESS:
            tqdm.write(f"\r{time_prefix()} Saving checkpoint...")
            import time

            time.sleep(5)
            os.makedirs(self.checkpoint_path, exist_ok=True)

        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(self.checkpoint_path, safe_serialization=self.safe_serialization)

        loss_tracker_path = os.path.join(self.checkpoint_path, TRAIN_LOSS_STATE_FILE)
        self.train_loss_state.save(loss_tracker_path)
        if MASTER_PROCESS:
            training_state_dict = self.state.to_dict()
            training_state_dict["epoch"] = epoch
            training_state_dict["train_step"] = train_step
            training_state_dict["global_step"] = global_step
            training_state_dict["evaluations_done"] = evaluations_done
            training_state_dict["finished"] = finished

            training_state_path = os.path.join(self.checkpoint_path, STATE_FILE)
            self.state.save(training_state_path, training_state_dict)
            tqdm.write(f"\033[A\033[K{time_prefix()} Checkpoint saved.")

    def epoch_iterator(self):
        """Epoch iterator handling logic for checkpointing."""
        start = self.state.epoch

        for epoch in range(start, self.hps.epochs):
            self.state.epoch = epoch
            if self.state.global_step % self.log_every:
                self.monitor.log_epoch(epoch)
            self.state.is_end_of_epoch = False
            self.state.is_last_epoch = epoch == self.hps.epochs - 1

            self.callback.on_epoch_start()
            yield epoch
            self.callback.on_epoch_end()

            if self._checkpointing_when_epoch_ends and (self.state.epoch + 1) % self.checkpoint_every == 0:
                self._save_checkpoint(
                    self.state.epoch + 1,
                    self.state.train_step,  # always 0 at this stage
                    self.state.global_step,
                    self.state.evaluations_done,
                    # flag as finished if checkpointing at the end of the last epoch
                    finished=self.state.is_last_epoch,
                )

    def _prepare(
        self,
        module: AcceleratorModule,
        model: nn.Module,
        teacher: Optional[nn.Module],
        train_dataloader: DataLoader,
        val_dataloader: Optional[dict[Any, DataLoader]],
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler],
        batch_device_placement: bool = True,
    ) -> tuple[nn.Module, Optional[nn.Module], DataLoader, Optional[DataLoader], Optimizer, Optional[LRScheduler]]:
        """
        Call Accelerate's backend to prepare instances for distributed training. This will also load states for objects
        in case of resuming training.
        """
        if self.compile and DEBUG_MODE < 2:
            model = torch.compile(model, **self.compile_kwargs)
            if teacher is not None:
                teacher = torch.compile(teacher, **self.compile_kwargs)

        if val_dataloader is not None:
            for k, dataloader in val_dataloader.items():
                val_dataloader[k] = self.accelerator.prepare_data_loader(dataloader)

        if self.accelerator.distributed_type == DistributedType.FSDP:
            # ignore model preparation since it was already done before (only in the case of FSDP)
            train_dataloader, optimizer, scheduler = self.accelerator.prepare(train_dataloader, optimizer, scheduler)
        else:
            model, train_dataloader, optimizer, scheduler = self.accelerator.prepare(
                model, train_dataloader, optimizer, scheduler
            )

        if self.accelerator.distributed_type != DistributedType.DEEPSPEED and teacher is not None:
            teacher = self.accelerator.prepare_model(teacher)

        if self.safe_mode or self.accelerator.distributed_type == DistributedType.FSDP:
            module.model = model
            if self.accelerator.distributed_type == DistributedType.MULTI_GPU:
                module.model = _DistributedDataParallel(module.model, model)

        if scheduler is not None:
            self.accelerator.register_for_checkpointing(scheduler)

        # load states if resuming
        if self.resume:
            self.callback.on_resume()
            if os.path.exists(self.checkpoint_path):
                self.accelerator.load_state(self.checkpoint_path)
            else:
                self.accelerator.end_training()
                raise FileNotFoundError(f"'{self.checkpoint_path}' was not found.")

        if not batch_device_placement:
            cpu = torch.device("cpu")
            train_dataloader.device = cpu
            for k in val_dataloader.keys():
                val_dataloader[k].device = cpu

        return model, teacher, train_dataloader, val_dataloader, optimizer, scheduler

    def _get_optimizer(self, module: AcceleratorModule) -> Optimizer:
        """Get optimizer from either module or trainer."""
        optimizer = module.get_optimizer()
        if optimizer is None:
            optimizer = self.hps.optimizer
            fused_available = "fused" in inspect.signature(optimizer).parameters
            optim_kwargs = self.hps.optim_kwargs
            optim_kwargs["fused"] = fused_available and "cuda" in self.accelerator.device.type
            filtered_kwargs = filter_kwargs(optim_kwargs, optimizer)

            optimizer = optimizer(module.model.parameters(), **filtered_kwargs)

        return optimizer

    def _get_scheduler(
        self, module: AcceleratorModule, optimizer: Optimizer, num_training_steps: int, num_epochs: int
    ) -> Optional[LRScheduler]:
        """Get scheduler from either module or trainer."""
        scheduler = module.get_scheduler(optimizer, num_training_steps, num_epochs)
        if self.hps.scheduler is not None and scheduler is None:
            schlr_kwargs = self.hps.scheduler_kwargs
            schlr_kwargs["last_epoch"] = -1
            schlr_kwargs["steps_per_epoch"] = num_training_steps
            total_steps = num_training_steps * num_epochs
            schlr_kwargs["num_training_steps"] = total_steps
            schlr_kwargs["epochs"] = num_epochs
            if "num_warmup_steps" in schlr_kwargs and isinstance(schlr_kwargs["num_warmup_steps"], float):
                if schlr_kwargs["num_warmup_steps"] < 0.0 or schlr_kwargs["num_warmup_steps"] > 1.0:
                    self.accelerator.end_training()
                    raise ValueError(
                        "If 'num_warmup_steps' is a ratio (float value), it needs to be a value between 0 and 1."
                    )
                schlr_kwargs["num_warmup_steps"] = round(total_steps * schlr_kwargs["num_warmup_steps"])
            elif "warmup_ratio" in schlr_kwargs:
                if schlr_kwargs["warmup_ratio"] > 1.0:
                    self.accelerator.end_training()
                    raise ValueError(
                        "'warmup_ratio' value in scheduler configuration needs to be a value between 0 and 1."
                    )
                schlr_kwargs["num_warmup_steps"] = round(total_steps * schlr_kwargs["warmup_ratio"])

            scheduler = self.hps.scheduler
            filtered_kwargs = filter_kwargs(schlr_kwargs, scheduler)

            scheduler = scheduler(optimizer, **filtered_kwargs)

        return scheduler

    def _get_dataloaders(
        self,
        module: AcceleratorModule,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Union[list[Dataset], dict[Any, Dataset]]] = None,
    ) -> tuple[DataLoader, Optional[dict[Any, DataLoader]]]:
        """Get DataLoaders for training and validation. Validation dataloaders will be wrapped in a dictionary."""
        is_tuple = hasattr(self.hps.batch_size, "__len__")
        if is_tuple and len(self.hps.batch_size) != 2:
            self.accelerator.end_training()
            raise ValueError(
                "'batch_size' in hyper parameters needs to be an integer value or a tuple with 2 values "
                "(one for training and the other for validation)."
            )

        train_batch_size = self.hps.batch_size[0] if is_tuple else self.hps.batch_size
        val_batch_size = self.hps.batch_size[1] if is_tuple else self.hps.batch_size

        dl_args = {
            "pin_memory": self.dataloader_pin_memory,
            "num_workers": self.dataloader_num_workers,
            "drop_last": self.dataloader_drop_last,
        }

        train_dataloader = module.get_train_dataloader()
        assert train_dataloader is not None or train_dataset is not None, (
            "Either 'train_dataset' or 'get_train_dataloader' must be given."
        )

        # ignoring 'train_dataset' if 'get_train_dataloader' was implemented in AcceleratorModule
        if train_dataset is not None and train_dataloader is None:
            shuffle_train = self.shuffle_train if self.sampler is None else None
            train_dataloader = DataLoader(
                train_dataset,
                shuffle=shuffle_train,
                sampler=self.samplers,
                batch_size=train_batch_size,
                collate_fn=self.collate_fn_train,
                **dl_args,
            )

        val_dataloader = module.get_validation_dataloader()
        if val_dataloader is not None and not isinstance(val_dataloader, (list, dict)):
            val_dataloader = [val_dataloader]

        # ignoring 'val_dataset' if 'get_validation_dataloader' was implemented in AcceleratorModule
        if val_dataset is not None and val_dataloader is None:
            val_dataset = (
                val_dataset if isinstance(val_dataset, dict) else {str(i): ds for i, ds in enumerate(val_dataset)}
            )
            val_dataloader = {}
            for k, dataset in val_dataset.items():
                val_dataloader[k] = DataLoader(
                    dataset,
                    batch_size=val_batch_size,
                    collate_fn=self.collate_fn_val,
                    **dl_args,
                )

        return train_dataloader, val_dataloader

    def _get_module(
        self, module: Union[AcceleratorModule, str, Union[tuple[str, str], tuple[str, Any]]], **kwargs: Any
    ) -> AcceleratorModule:
        """Get module corresponding to the arguments given."""
        if isinstance(module, str):
            return AcceleratorModule.from_hf(module, **kwargs)
        elif isinstance(module, tuple):
            return AcceleratorModule.from_hf(*module, **kwargs)

        return module

    def _init_trackers(self):
        """Initialize all trackers along with the training configuration from Hyper Parameters and 'additional_tracker_config'."""
        self.accelerator.log_with = [tracker.tracker for tracker in self.log_with]
        track_name = os.path.basename(self.model_path) if self.track_name is None else self.track_name
        init_kwargs = combine_dicts(*[tracker.init(**self.init_kwargs) for tracker in self.log_with])

        config = self.hps.get_config()
        effective_num = self.grad_accumulation_steps * self.accelerator.num_processes
        config["effective_batch_size"] = (
            tuple(batch_size * effective_num for batch_size in self.hps.batch_size)
            if isinstance(self.hps.batch_size, (tuple, list))
            else self.hps.batch_size * effective_num
        )
        config["grad_accumulation_steps"] = self.grad_accumulation_steps
        config["num_processes"] = self.accelerator.num_processes

        tracker_config = config | self.additional_tracker_config

        if MASTER_PROCESS:
            # TODO with a Tracker Wrapper this should be fixed.
            _is_url = is_url(self.logging_dir)
            if _is_url and not self._logging:
                self.accelerator.end_training()
                raise RuntimeError(f"Cannot log results in '{self.logging_dir}' because 'log_with' was not declared.")

            self.accelerator.init_trackers(track_name, config=tracker_config, init_kwargs=init_kwargs)
            for logger in self.log_with:
                if logger.tracker == LoggerType.MLFLOW and _is_url:
                    import mlflow

                    mlflow.set_tracking_uri(self.logging_dir)

    def _get_grad_norm(self, model: nn.Module) -> float:
        """Calculates grad norm of model."""
        return math.sqrt(sum([torch.norm(p.grad.detach()) ** 2 for p in model.parameters() if p.grad is not None]))

    def log_artifact(self, path: str, **kwargs: Any):
        """
        Logs an artifact to the current run. **NOTE**: Current implementation only works for MLFlow.

        Args:
            path (`str`):
                Path to the file to be logged as an artifact.
            kwargs (`Any`):
                Extra arguments for tracker's log_artifact function.
        """
        # TODO incorporate this functionality in a Tracker Wrapper.
        if MASTER_PROCESS and DEBUG_MODE < 1:
            for logger in self.log_with:
                if logger.tracker == LoggerType.MLFLOW:
                    import mlflow

                    mlflow.log_artifact(path, **kwargs)
                else:
                    self.accelerator.end_training()
                    raise NotImplementedError("'log_artifact' is only supported for MLFlow (for now).")

    def _prepare_metrics(
        self,
        metrics: Union[Metric, list[Metric], dict[Any, Union[Metric, list[Metric]]]],
        val_dataloader: Optional[dict[Any, DataLoader]],
    ) -> dict[Any, list[Metric]]:
        """Prepare metrics in relation to validation datasets, running checks for types and fixing them if possible."""
        if isinstance(metrics, Metric):
            metrics = {k: [metrics] for k in val_dataloader.keys()}
        elif isinstance(metrics, list):
            metrics = {k: metrics for k in val_dataloader.keys()}
        elif isinstance(metrics, dict):
            assert all(k in val_dataloader for k in metrics), (
                f"There is a mismatch between given metrics and validation datasets. Got {list(metrics.keys())} "
                f"for 'metrics' and {list(val_dataloader.keys())} for validation datasets."
            )
            metrics = {k: (v if isinstance(v, list) else [v]) for k, v in metrics.items()}

        return metrics

    def register_model_saving(
        self,
        model_saving: str,
        saving_below: Optional[float] = None,
        saving_above: Optional[float] = None,
    ):
        """
        Register a type of model saving.

        Args:
            model_saving (`str`):
                Type of model saving. It can be `"best_valid_loss"` (default), `"best_train_loss"` or in format of
                `"best_{METRIC}"`. **NOTE**: `"best_"` is optional. Also, all metrics should relate directly to metrics
                and validation datasets. This can also be in the form of `"best_{METRIC}@{DATASET}"` (metric at a specific dataset),
                `"best_{METRIC}@{DATASET1}@{DATASET2}"` (metric at dataset1 and dataset2), `"best_{METRIC1}@{DATASET1}/{METRIC1}@{dataset2}"`
                (best metric1 at dataset1 and best metric2 at dataset2), `"best_{METRIC1}/{METRIC2}@{DATASET2}"` (best metric1 between all
                datasets containing this metric and best metric2 at dataset2 only), etc.
            saving_below (`float`, *optional*, defaults to `None`):
                Register this model saving to only be saved whenever its values are lower than this.
            saving_above (`float`, *optional*, defaults to `None`):
                Register this model saving to only be saved whenever its values are above than this.
        """
        saving_below = saving_below if saving_below is not None else float("inf")
        saving_above = saving_above if saving_above is not None else float("-inf")

        model_saving = f"best_{model_saving}" if not model_saving.startswith("best_") else model_saving

        self.model_saving[model_saving] = (saving_below, saving_above)

    def _get_comparator(self, metric: str) -> str:
        """Get comparator for a given metric."""
        for dataset_key, metrics in self.metrics.items():
            for _metric in metrics:
                if metric == _metric.main_metric:
                    return _metric.comparator

        self.accelerator.end_training()
        raise RuntimeError(f"No comparator was found for metric '{metric}'.")
