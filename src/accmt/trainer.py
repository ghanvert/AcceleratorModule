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
_default_model_savings = set({"best_valid_loss", "best_train_loss", "always"})
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
        model_saving: Optional[Union[str, list[str]]] = "best_valid_loss",
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
        model_saving_below: Optional[float] = None,
        model_saving_above: Optional[float] = None,
        collate_fn_train: Optional[Callable] = None,
        collate_fn_val: Optional[Callable] = None,
        max_shard_size: str = "10GB",
        safe_serialization: bool = False,
        compile: bool = False,
        train_loss_metric_name: str = "train_loss",
        val_loss_metric_name: str = "val_loss",
        dataloader_pin_memory: bool = True,
        dataloader_num_workers: Optional[int] = None,
        dataloader_drop_last: bool = False,
        eval_when_finish: bool = True,
        eval_when_start: bool = False,
        monitor: Optional[Monitor] = None,
        metrics: Optional[Union[Metric, list[Metric]]] = None,
        cleanup_cache_every_n_steps: Optional[int] = None,
        callback: Optional[Union[Callback, list[Callback]]] = None,
        additional_tracker_config: Optional[dict[str, Any]] = None,
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
            model_saving (`str` or `list`, *optional*, defaults to `best_valid_loss`):
                Type of model saving. It can be one of the following values:

                - `"best_valid_loss"`: Saves the model whenever the validation loss is the best recorded.
                - `"best_train_loss"`: Saves the model whenever the training loss is the best recorded.
                - `"always"`: Saves the model always at the end of every evaluation.
                - in format of `"best_{METRIC}"`, where METRIC corresponds to the additional metric in `additional_metrics`.

                If not specified (`None`), model saving will be disabled. This can also be a list of model savings methods to save
                more models on different metrics. When implementing multiple model savings, the resulting model path will be of the form
                "{model_path}_best_{metric}", like the following examples:
                    - MODEL_best_accuracy
                    - MODEL_best_train_loss
            patience (`int`, *optional*, defaults to `None`):
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

                    - `TensorBoard`
                    - `WandB`
                    - `CometML`
                    - `Aim`
                    - `MLFlow`
                    - `ClearML`
                    - `DVCLive`
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
            model_saving_below (`float`, *optional*, defaults to `None`):
                Start saving model below this metric (based on `model_saving`).
            model_saving_above (`float`, *optional*, defaults to `None`):
                Start saving model above this metric (based on `model_saving`).
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
            metrics (`list`, *optional*, defaults to `None`):
                List of additional metrics of type 'Metric' to track.
            cleanup_cache_every_n_steps (`int`, *optional*, defaults to `None`):
                Cleanup CPU and CUDA caches every N steps. Default is no cleanup.

                NOTE: On every epoch and evaluation call we cleanup cache.
            callback (`Callback` or `list`, *optional*, defaults to `None`):
                `Callback` or callbacks to implement.
            additional_tracker_config (`dict`, *optional*, defaults to `None`):
                Additional configuration specification for tracker (e.g. hyper-parameters).
            kwargs (`Any`, *optional*):
                Extra arguments for specific `init` function in Tracker, e.g. `run_name`, `tags`, etc.
        """
        # do some previous checks
        self.model_saving = model_saving if isinstance(model_saving, list) else [model_saving]
        self.metrics = metrics if isinstance(metrics, list) else [metrics]
        self.log_with = log_with if isinstance(log_with, list) else [log_with]
        self.log_with = [tracker for tracker in self.log_with if tracker is not None]
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

        self.metrics = [metric for metric in self.metrics if metric is not None]

        # create temporary set of metrics with prefix "best_" to check if metrics are correctly alligned with model savings
        _implemented_metrics = _default_model_savings | set(
            f"best_{metric.main_metric}" if not metric.main_metric.startswith("best_") else metric.main_metric
            for metric in self.metrics
        )
        self.model_saving = [ms.lower() for ms in self.model_saving]
        self.model_saving = [f"best_{ms}" if not ms.startswith("best_") else ms for ms in self.model_saving]
        assert all(ms in _implemented_metrics for ms in self.model_saving), (
            f"All 'model_saving' methods should be declared in 'metrics' or be one of {_default_model_savings}."
        )

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
        if clip_grad is not None and self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            rprint(
                "[WARNING] Clipping gradient using Trainer is not supported when running with DeepSpeed. Setting it to None."
            )
            clip_grad = None
        self.clip_grad = clip_grad
        self.set_to_none = set_to_none
        self.shuffle_train = shuffle_train
        self.sampler = sampler
        self.model_saving_below = model_saving_below if model_saving_below is not None else float("inf")
        self.model_saving_above = model_saving_above if model_saving_above is not None else float("-inf")
        self.collate_fn_train = collate_fn_train
        self.collate_fn_val = collate_fn_val
        self.max_shard_size = max_shard_size
        self.safe_serialization = safe_serialization
        self.compile = compile
        self.train_loss_metric_name = train_loss_metric_name
        self.val_loss_metric_name = val_loss_metric_name
        self.dataloader_pin_memory = dataloader_pin_memory if IS_GPU else False
        self.dataloader_num_workers = (
            dataloader_num_workers if dataloader_num_workers is not None else self.accelerator.num_processes
        )
        self.dataloader_drop_last = dataloader_drop_last
        self.samplers = sampler
        self.eval_when_finish = eval_when_finish
        self.eval_when_start = eval_when_start if DEBUG_MODE < 4 else False
        self.monitor = monitor if isinstance(monitor, Monitor) else Monitor.from_config(monitor)
        self.monitor.grad_norm = (
            self.monitor.grad_norm if self.accelerator.distributed_type == DistributedType.DEEPSPEED else False
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
        self.init_kwargs = kwargs

        self.accelerator.project_configuration = ProjectConfiguration(
            project_dir=".", logging_dir=logging_dir, total_limit=1
        )

        self._logging = len(self.log_with) > 0
        if self._logging and DEBUG_MODE < 1:
            self._init_trackers()

        self.state = TrainingState()
        self.state.patience_left = {ms: self.patience for ms in self.model_saving}
        self.gatherer = Gatherer()
        # adding a total (at maximum) of 64 bytes for additional tensors
        self.train_loss_state = LossState(self.accelerator, self.accelerator.device, self.log_every)
        self.val_loss_state = LossState(self.accelerator, self.accelerator.device, -1)

        self._checkpointing_every_n_steps = self.enable_checkpointing and self.checkpoint_strat == "step"
        self._checkpointing_after_evaluation = self.enable_checkpointing and self.checkpoint_strat == "eval"
        self._checkpointing_when_epoch_ends = self.enable_checkpointing and self.checkpoint_strat == "epoch"

    def fit(
        self,
        module: Union[AcceleratorModule, str, Union[tuple[str, str], tuple[str, Any]]],
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
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
            val_dataset (`torch.utils.data.Dataset`, *optional*, defaults to `None`):
                `Dataset` class from PyTorch containing the validation dataset logic. If this
                dataset is not specified, then the validation logic of `AcceleratorModule`
                (if specified) will be skipped. If `model_saving` parameter in the constructor is set
                to `best_valid_loss`, this will be converted to `best_train_loss` in the background.
                If not provided, it will use `get_validation_dataloader` to get the validation DataLoader
                (if implemented).
            kwargs (`Any`):
                Keyword arguments for `from_pretrained` function for model initialization.
        """
        # reset loss states in case of another fit function call in the script
        self.train_loss_state.reset()
        self.val_loss_state.reset()

        module = self._get_module(module, **kwargs)
        module._log_every = self.log_every
        module.batch_size = self.hps.batch_size

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

        if self.compile and DEBUG_MODE < 2:
            model = torch.compile(model)
            if teacher is not None:
                teacher = torch.compile(teacher)

        if MASTER_PROCESS and DEBUG_MODE < 3:
            os.makedirs(self.model_path, exist_ok=True)

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

        train_dataloader, val_dataloader = self._get_dataloaders(module, train_dataset, val_dataset)

        optimizer = self._get_optimizer(module)
        scheduler = self._get_scheduler(
            module, optimizer, round(len(train_dataloader) / self.accelerator.num_processes), self.hps.epochs
        )

        model, teacher, train_dataloader, val_dataloader, optimizer, scheduler = self._prepare(
            module, model, teacher, train_dataloader, val_dataloader, optimizer, scheduler
        )

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
        val_dataloader: Optional[DataLoader],
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler],
    ):
        """Runs a training loop."""
        if self.state.evaluations_done == 0 and self.eval_when_start:
            ""
            self.eval(module, model, val_dataloader)

        for epoch in self.epoch_iterator():
            for batch in self.batch_iterator(train_dataloader, model):
                if (self.state.global_step + 1) % self.evaluate_every_n_steps == 0:
                    self.eval(module, model, val_dataloader)

                self._train_logic(module, model, optimizer, batch, scheduler)

            if self.evaluate_every_n_steps is None or (self.eval_when_finish and epoch == self.hps.epochs - 1):
                self.eval(module, model, val_dataloader)

    @torch.inference_mode()
    def eval(self, module: AcceleratorModule, model: nn.Module, dataloader: Optional[DataLoader]):
        """Runs evaluation on a given dataloader."""
        if DEBUG_MODE >= 5:
            return

        if model.training:
            model.eval()

        cleanup()
        self.callback.on_evaluation_start()
        # do evaluation if available
        if dataloader is not None:
            for i, batch in tqdm(
                iterable=enumerate(dataloader),
                total=len(dataloader),
                desc=f"📊 Evaluating in Epoch {self.state.epoch + 1}/{self.hps.epochs}",
                position=1,
                colour="cyan",
                **_tqdm_kwargs,
            ):
                self.state.val_step = i
                self._validation_logic(module, batch)

        # only master process will be in charge of calculating metrics to avoid system overhead
        if MASTER_PROCESS:
            for metric in self.metrics:
                metric_dict = metric._compute()

                for m, v in metric_dict.items():
                    if not isinstance(v, (float, int, torch.Tensor)):
                        self.accelerator.end_training()
                        raise ValueError(
                            f"Value in metric's dict does not accept {type(v)}, only "
                            f"`float`, `int`, `torch.Tensor` (torch) or `NDArray` (numpy)"
                        )

                    self.state.additional_metrics[m] = v if not isinstance(v, (torch.Tensor, np.ndarray)) else v.item()

        self.state.evaluations_done += 1
        self.callback.on_evaluation_end()

        # save model
        if self.model_saving is not None and DEBUG_MODE < 3:
            self._save_model_on_criteria(model)
        else:
            # reset total loss state for validation since it's not being used
            self.val_loss_state.total_loss.zero_()
            self.val_loss_state.steps.zero_()

        if self._checkpointing_after_evaluation:
            self._save_checkpoint(self.state.epoch, self.state.train_step + 1)

        self.state.val_step = 0

    def _save_model_on_criteria(self, model: nn.Module):
        """Save model depending on criteria defined in `model_saving`"""
        self.accelerator.wait_for_everyone()

        train_loss = self.train_loss_state.get_total_loss()
        val_loss = self.val_loss_state.get_total_loss()

        saving_criteria = {"always": True}
        for metric in self.metrics:
            best_metric_str = f"best_{metric.main_metric}"
            if best_metric_str in self.state.additional_metrics:
                prev = self.state.additional_metrics[best_metric_str]
                new = self.state.additional_metrics[metric.main_metric]
                compare = operator_map[metric.comparator]
                is_better = compare(new, prev)  # e.g. new > prev
                best = new if is_better else prev
            else:
                start_value = float("inf") if metric.comparator in {"<", "<=", "=="} else float("-inf")
                new = start_value
                is_better = False
                best = new

            self.state.additional_metrics[best_metric_str] = best
            saving_criteria[best_metric_str] = (
                is_better and new < self.model_saving_below and new > self.model_saving_above
            )

        saving_criteria["best_valid_loss"] = (
            val_loss < self.state.best_valid_loss
            and val_loss < self.model_saving_below
            and val_loss > self.model_saving_above
        )
        self.state.best_valid_loss = val_loss if val_loss < self.state.best_valid_loss else self.state.best_valid_loss

        # ignore first track of training loss when evaluating at the start (since train loss does not exist at this stage)
        if not (self.state.evaluations_done == 1 and self.eval_when_start):
            saving_criteria["best_train_loss"] = (
                train_loss < self.state.best_train_loss
                and train_loss < self.model_saving_below
                and train_loss > self.model_saving_above
            )
            self.state.best_train_loss = (
                train_loss if train_loss < self.state.best_train_loss else self.state.best_train_loss
            )

        for model_saving in self.model_saving:
            if saving_criteria[model_saving] and self.state.patience_left[model_saving] != 0:
                if MASTER_PROCESS:
                    model_path = os.path.join(self.model_path, model_saving)
                    self._save_model(model, model_path)
            else:
                if self.state.patience_left[model_saving] > 0 and not (
                    self.eval_when_start and self.state.evaluations_done == 1
                ):
                    self.state.patience_left[model_saving] -= 1

        # count model savings with patience_left equal 0
        count = 0
        for model_saving in self.model_saving:
            if self.state.patience_left[model_saving] == 0:
                count += 1

        # if all model savings have no patience anymore, finish training process
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

    def _validation_logic(self, module: AcceleratorModule, batch: Any):
        """Runs all the validation logic."""
        self.callback.on_before_validation_step(batch)
        metrics = module.validation_step(batch)
        self.callback.on_after_validation_step()
        # track loss
        loss = metrics["loss"].detach()
        self.val_loss_state.add_total_loss(loss)

        # track metrics
        for metric in self.metrics:
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

            if (self.state.global_step + 1) % self.log_every == 0:
                batch_loss = self.train_loss_state.get_batch_loss()

                norm = None
                if MASTER_PROCESS and self.monitor.grad_norm:
                    norm = self._get_grad_norm()

                self.monitor.log_train_loss_and_grad_norm(batch_loss, norm)

            if not module._extended:
                # backpropagation
                self.callback.on_before_backward(loss)
                self.accelerator.backward(loss)
                self.callback.on_after_backward()

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
            self.state.global_step = i
            yield batch

            if (
                self.cleanup_cache_every_n_steps is not None
                and (self.state.global_step + 1) % self.cleanup_cache_every_n_steps == 0
            ):
                cleanup()

            if self._checkpointing_every_n_steps and (self.state.global_step + 1) % self.checkpoint_every == 0:
                self._save_checkpoint(self.state.epoch, self.state.train_step + 1)

        self.state.train_step = 0

    def _save_checkpoint(self, epoch: int, train_step: int):
        """Save checkpoint at a given point in time (`epoch` and `train_step`)."""
        self.callback.on_save_checkpoint()
        if MASTER_PROCESS:
            tqdm.write(f"\r{time_prefix()} Saving checkpoint...")
            import time

            time.sleep(5)
            os.makedirs(self.checkpoint_path, exist_ok=True)

        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(self.checkpoint_path, safe_serialization=self.safe_serialization)

        if MASTER_PROCESS:
            training_state_dict = self.state.to_dict()
            training_state_dict["epoch"] = epoch
            training_state_dict["train_step"] = train_step

            training_state_path = os.path.join(self.checkpoint_path, STATE_FILE)
            loss_tracker_path = os.path.join(self.checkpoint_path, TRAIN_LOSS_STATE_FILE)
            self.state.save(training_state_path, training_state_dict)
            self.train_loss_state.save(loss_tracker_path)
            tqdm.write(f"\033[A\033[K{time_prefix()} Checkpoint saved.")

    def epoch_iterator(self):
        """Epoch iterator handling logic for checkpointing."""
        start = self.state.epoch

        for epoch in range(start, self.hps.epochs):
            self.state.epoch = epoch

            self.callback.on_epoch_start()
            yield epoch
            self.callback.on_epoch_end()

            if self._checkpointing_when_epoch_ends and (self.state.epoch + 1) % self.checkpoint_every == 0:
                self._save_checkpoint(self.state.epoch + 1, 0)

    def _prepare(
        self,
        module: AcceleratorModule,
        model: nn.Module,
        teacher: Optional[nn.Module],
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader],
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler],
    ) -> tuple[nn.Module, Optional[nn.Module], DataLoader, Optional[DataLoader], Optimizer, Optional[LRScheduler]]:
        """
        Call Accelerate's backend to prepare instances for distributed training. This will also load states for objects
        in case of resuming training.
        """
        if self.accelerator.distributed_type == DistributedType.FSDP:
            # ignore model preparation since it was already done before (only in the case of FSDP)
            train_dataloader, val_dataloader, optimizer, scheduler = self.accelerator.prepare(
                train_dataloader, val_dataloader, optimizer, scheduler
            )
        else:
            model, train_dataloader, val_dataloader, optimizer, scheduler = self.accelerator.prepare(
                model, train_dataloader, val_dataloader, optimizer, scheduler
            )

        if self.accelerator.distributed_type != DistributedType.DEEPSPEED and teacher is not None:
            teacher = self.accelerator.prepare_model(teacher)

        if self.accelerator.distributed_type == DistributedType.FSDP:
            module.model = model  # force module.model to be wrapped to not have problems with dimensions

        # NOTE: we aren't forcing module.model to be wrapped for other distributed types since we haven't seen any
        # issues with training, and it's actually a little bit faster doing inference with the model directly on the Module class.

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

        return model, teacher, train_dataloader, val_dataloader, optimizer, scheduler

    def _get_optimizer(self, module: AcceleratorModule) -> Optimizer:
        """Get optimizer from either module or trainer."""
        optimizer = module.get_optimizer()
        if optimizer is None:
            optimizer = self.hps.optim
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
        val_dataset: Optional[Dataset] = None,
    ) -> tuple[DataLoader, Optional[DataLoader]]:
        """Get DataLoaders for training and validation."""
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
        # ignoring 'val_dataset' if 'get_validation_dataloader' was implemented in AcceleratorModule
        if val_dataset is not None and val_dataloader is None:
            val_dataloader = DataLoader(
                val_dataset,
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
        self.accelerator.init_trackers(track_name, config=tracker_config, init_kwargs=init_kwargs)

        if MASTER_PROCESS:
            # TODO with a Tracker Wrapper this should be fixed.
            for logger in self.log_with:
                if logger.tracker == LoggerType.MLFLOW and is_url(self.logging_dir):
                    import mlflow

                    mlflow.set_tracking_uri(self.logging_dir)

    def _get_grad_norm(self, model: nn.Module) -> float:
        """Calculates grad norm of model."""
        return math.sqrt(
            sum([torch.norm(p.grad.detach()) ** 2 for p in model.parameters() if p.grad is not None]).item()
        )

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
