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
import traceback
from typing import Optional, Union

import torch
import torch.nn as nn
from accelerate import DistributedType
from accelerate.utils import LoggerType, ProjectConfiguration, set_seed, tqdm
from torch.utils.data import DataLoader, Dataset
from typing_extensions import Any

from .callbacks import Callback
from .dataloader_samplers import BaseSampler
from .dist_utils import Gatherer
from .handlers import Handler
from .hyperparameters import HyperParameters
from .metrics import Metric
from .modules import AcceleratorModule
from .monitor import Monitor
from .states import TrainingState
from .utils import (
    cleanup,
    combine_dicts,
    get_number_and_unit,
    is_url,
    operator_map,
    read_status,
    save_status,
    time_prefix,
)


CHECKPOINT_PATH = "checkpoint"
STATUS_PATH = "status.json"
DEBUG_MODE: int = int(os.environ.get("ACCMT_DEBUG_MODE", 0))


class Trainer:
    """
    Class to implement the training configuration.
    """

    @classmethod
    def from_config(
        cls,
        config: Union[str, dict],
        log_with: Optional[Union[Any, list]] = None,
        sampler: Optional[Union[Any, list]] = None,
        collate_fn_train: Optional[Any] = None,
        collate_fn_val: Optional[Any] = None,
    ):
        """
        Load a configuration from a file or a dictionary.

        Args:
            config (`str` or `dict`):
                Path to a file or dictionary containing kwargs for Trainer constructor. The file can
                be YAML or JSON.
            log_with (`accmt.tracker` or `list`, *optional*, defaults to `None`):
                Logger to log metrics.
            sampler (list or `Any`, *optional*, defaults to `None`):
                Sampler (or list of samplers) for train DataLoader.
            collate_fn_train (`function` or `list`, *optional*, defaults to `None`):
                Collate function to be implemented in train dataloader.
            collate_fn_val (`function` or `list`, *optional*, defaults to `None`):
                Collate function to be implemented in validation dataloader.
        """
        assert isinstance(config, (str, dict)), "'config' needs to be either a path to a file, or a dictionary."
        if isinstance(config, str):
            import yaml

            config = yaml.safe_load(open(config))

        return Trainer(
            **config,
            log_with=log_with,
            sampler=sampler,
            collate_fn_train=collate_fn_train,
            collate_fn_val=collate_fn_val,
        )

    def __init__(
        self,
        hps_config: Union[str, dict, HyperParameters],
        model_path: str,
        track_name: Optional[str] = None,
        checkpoint: Optional[str] = None,
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
        shuffle_validation: bool = False,
        sampler: Optional[Union[Any, list]] = None,
        model_saving_below: Optional[float] = None,
        model_saving_above: Optional[float] = None,
        collate_fn_train: Optional[Any] = None,
        collate_fn_val: Optional[Any] = None,
        disable_collate_fn_on_evaluation: bool = False,
        max_shard_size: str = "10GB",
        safe_serialization: bool = False,
        compile: bool = False,
        train_loss_metric_name: str = "train_loss",
        val_loss_metric_name: str = "val_loss",
        dataloader_pin_memory: bool = True,
        dataloader_num_workers: Optional[int] = None,
        dataloader_drop_last: bool = False,
        handlers: Optional[Union[list, Any]] = None,
        eval_when_finish: bool = True,
        eval_when_start: bool = False,
        monitor: Optional[Monitor] = None,
        metrics: Optional[Union[Metric, list[Metric]]] = None,
        cleanup_cache_every_n_steps: Optional[int] = None,
        callback: Optional[Callback] = None,
        additional_tracker_config: Optional[dict[str, Any]] = None,
        report_train_loss_per_epoch: bool = False,
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
            checkpoint (`str`, *optional*, defaults to `None`):
                Path where to save the checkpoint. Path by default is going to be of type:
                'checkpoint-MODEL_PATH_NAME'.
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
                Log every N steps. If `grad_accumulation_steps` is set to a higher value than `1`, then this parameter will be
                modified to be `log_every` * `grad_accumulation_steps` ONLY when logging during training.
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
            shuffle_validation (`bool`, *optional*, defaults to `False`):
                Whether to shuffle validation DataLoader.
            sampler (`list` or `Any`, *optional*, defaults to `None`):
                Sampler (or list of samplers) for train DataLoader.
            model_saving_below (`float`, *optional*, defaults to `None`):
                Start saving model below this metric (based on `model_saving`).
            model_saving_above (`float`, *optional*, defaults to `None`):
                Start saving model above this metric (based on `model_saving`).
            collate_fn_train (`function` or `list`, *optional*, defaults to `None`):
                Collate function to be implemented in train dataloader. If `module` overrides `collate_fn` from
                `AcceleratorModule` class, then that function will be used instead of the one specified on
                this constructor. If a list of collate functions is given, then the every collate function will affect
                the batch in the given order.
            collate_fn_val (`function` or `list`, *optional*, defaults to `None`):
                Collate function to be implemented in validation dataloader.
            disable_collate_fn_on_evaluation (`bool`, *optional*, defaults to `False`):
                Disable 'collate_fn' on validation DataLoader.
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
            handlers (`Any` or `list`, *optional*, defaults to `None`):
                Handler or List of handlers to catch errors and make a safe checkpoint.
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

                NOTE: Every epoch and every evaluation call we cleanup cache.
            callback (`Callback`, *optional*, defaults to `None`):
                `Callback` to implement. This module will have the logic for every existing callback function.
            additional_tracker_config (`dict`, *optional*, defaults to `None`):
                Additional configuration specification for tracker (e.g. hyper-parameters).
            report_train_loss_per_epoch (`bool`, *optional*, defaults to `False`):
                Report train loss at the end of every epoch (instead of `log_every`).
            kwargs (`Any`, *optional*):
                Extra arguments for specific `init` function in Tracker, e.g. `run_name`, `tags`, etc.
        """
        assert isinstance(hps_config, (str, dict, HyperParameters)), (
            "'hps_config' needs to be either a string, dictionary or HyperParameters class."
        )
        from . import IS_GPU, accelerator

        self.is_gpu = IS_GPU
        self.accelerator = accelerator
        self.hps = HyperParameters.from_config(hps_config) if isinstance(hps_config, (str, dict)) else hps_config
        self.track_name = track_name
        self.checkpoint = checkpoint if checkpoint is not None else "checkpoint"
        self.checkpoint = f"{model_path}/{self.checkpoint}"
        self.resume = (
            resume if resume is not None else os.path.exists(self.checkpoint) and len(os.listdir(self.checkpoint)) > 0
        )
        if DEBUG_MODE >= 3:
            self.resume = False
        self.model_path = model_path
        self.metrics = metrics if isinstance(metrics, list) else [metrics]
        self.metrics = [metric for metric in self.metrics if metric is not None]

        _default_model_savings = set({"best_valid_loss", "best_train_loss", "always"})
        _implemented_metrics = set(
            f"best_{metric.main_metric}" if not metric.main_metric.startswith("best_") else metric
            for metric in self.metrics
        )
        _implemented_metrics.update(_default_model_savings)
        self.model_saving = model_saving if isinstance(model_saving, list) else [model_saving]
        self.model_saving = [ms.lower() for ms in self.model_saving]
        self.model_saving = [f"best_{ms}" if not ms.startswith("best_") else ms for ms in self.model_saving]
        assert all(ms in _implemented_metrics for ms in self.model_saving), (
            f"All 'model_saving' methods should be declared in 'metrics' or be one of {_default_model_savings}."
        )
        self.patience = patience if patience is not None else -1
        if self.patience == 0:
            raise ValueError("The `patience` argument in Trainer should have a value greater than 0.")

        self.evaluate_every_n_steps = evaluate_every_n_steps
        self.checkpoint_every = checkpoint_every
        if self.checkpoint_every is not None:
            self.checkpoint_every, self.checkpoint_strat = get_number_and_unit(self.checkpoint_every)
            self.enable_checkpointing = True
        else:
            # fix invalid arguments
            self.checkpoint_every = 1
            self.checkpoint_strat = "epoch"
            self.enable_checkpointing = False

        self.logging_dir = logging_dir
        self.log_with = None
        self.log_every = log_every
        self.grad_accumulation_steps = grad_accumulation_steps if grad_accumulation_steps is not None else 1
        assert clip_grad is None or isinstance(clip_grad, float), "'clip_grad' argument needs to be a float."
        if clip_grad is not None and self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            self.accelerator.print(
                time_prefix(),
                "[WARNING] Clipping gradient using Trainer is not supported when running with DeepSpeed. Setting it to None.",
            )
            clip_grad = None
        self.clip_grad = clip_grad
        self.set_to_none = set_to_none
        self.shuffle_train = shuffle_train
        self.shuffle_validation = shuffle_validation
        self.sampler = sampler
        self.model_saving_below = model_saving_below if model_saving_below is not None else float("inf")
        self.model_saving_above = model_saving_above if model_saving_above is not None else float("-inf")
        self.collate_fn_train = (
            self._get_collate_fn_pipeline(collate_fn_train) if isinstance(collate_fn_train, list) else collate_fn_train
        )
        self.collate_fn_val = (
            self._get_collate_fn_pipeline(collate_fn_val) if isinstance(collate_fn_val, list) else collate_fn_val
        )
        self.disable_collate_fn_on_evaluation = disable_collate_fn_on_evaluation
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
        self.handlers = handlers if isinstance(handlers, list) else [handlers]
        if self.handlers[0] is not None:
            raise NotImplementedError("'handlers' argument is not yet fully implemented.")
        self.eval_when_finish = eval_when_finish
        self.eval_when_start = eval_when_start if DEBUG_MODE < 4 else False
        self.monitor = monitor if isinstance(monitor, Monitor) else Monitor.from_config(monitor)
        self.monitor._debug_mode = DEBUG_MODE
        self.monitor.grad_norm = (
            self.monitor.grad_norm if self.accelerator.distributed_type == DistributedType.DEEPSPEED else False
        )
        if self.monitor.grad_norm and self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            self.accelerator.print(
                time_prefix(),
                "[WARNING] Gradient norm monitoring is not yet supported when running with DeepSpeed. Setting it to False.",
            )
            self.monitor.grad_norm = False
        self.cleanup_cache_every_n_steps = cleanup_cache_every_n_steps
        self.callback = callback if callback is not None else Callback()
        self.additional_tracker_config = additional_tracker_config if additional_tracker_config is not None else {}
        self.init_kwargs = kwargs

        self.accelerator.project_configuration = ProjectConfiguration(
            project_dir=".", logging_dir=logging_dir, total_limit=1
        )

        if log_with is not None:
            if not isinstance(log_with, list):
                log_with = [log_with]
            self.log_with = [tracker for tracker in log_with]
            if DEBUG_MODE < 1:
                self.accelerator.log_with = [tracker.tracker for tracker in log_with]

        self.report_train_loss_per_epoch = report_train_loss_per_epoch

        # we need to calculate mean for these tensors.
        self.train_total_loss: torch.Tensor = None  # loss tensor for evaluation
        self.train_track_loss: torch.Tensor = None  # train loss tensor to be reported
        self.val_total_loss: torch.Tensor = None  # val loss tensor for evaluation
        self.val_track_loss: torch.Tensor = None  # val loss tensor to be reported

        self.train_dataloader: DataLoader = None
        self.val_dataloader: DataLoader = None

        self.module = None

        self.gatherer = Gatherer()

        if self.log_with is not None:
            track_name = self.model_path.split("/")[-1] if self.track_name is None else self.track_name
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

            if DEBUG_MODE < 1:
                tracker_config = config | self.additional_tracker_config
                self.accelerator.init_trackers(track_name, config=tracker_config, init_kwargs=init_kwargs)

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
        self.module = module
        self.monitor.additional_metrics = self.metrics is not None and len(self.metrics) > 0
        self.monitor._do_tracking = self.log_with is not None
        self.train_total_loss = torch.tensor(0.0, device=self.accelerator.device, pin_memory=self.is_gpu)
        self.train_track_loss = torch.tensor(0.0, device=self.accelerator.device, pin_memory=self.is_gpu)
        self.val_total_loss = torch.tensor(0.0, device=self.accelerator.device, pin_memory=self.is_gpu)
        self.val_track_loss = torch.tensor(0.0, device=self.accelerator.device, pin_memory=self.is_gpu)

        if isinstance(module, str):
            module = AcceleratorModule.from_hf(module, **kwargs)
        elif isinstance(module, tuple):
            module = AcceleratorModule.from_hf(*module, **kwargs)

        model = getattr(module, "model", None)
        if model is None:
            raise AttributeError("'self.model' needs to be declared in the AcceleratorModule class.")
        elif model is not None and not isinstance(model, nn.Module):
            raise ValueError("'self.model' needs to be an instance of 'nn.Module'.")

        teacher = getattr(module, "teacher", None)
        if teacher is not None and not isinstance(teacher, nn.Module):
            raise ValueError("'self.teacher' needs to be an instance of 'nn.Module'.")

        module._log_every = self.log_every
        if torch.cuda.is_available():
            model.to(self.accelerator.device)  # for optimizer to apply fused when available
            if teacher is not None:
                teacher.eval()
                teacher.to(self.accelerator.device)
        if self.compile and DEBUG_MODE < 2:
            model = torch.compile(model)
            if teacher is not None:
                teacher = torch.compile(teacher)

        if self.accelerator.distributed_type == DistributedType.FSDP:
            # preparing model before dataloaders is only supported by FSDP apparently, and this is the
            # recommended setting to prepare training.
            model = self.accelerator.prepare(model)

        if self.accelerator.is_main_process and DEBUG_MODE < 3:
            os.makedirs(self.model_path, exist_ok=True)

        if self.resume:
            status_dict = read_status(f"{self.checkpoint}/{STATUS_PATH}")
            if "evaluations_done" not in status_dict:
                # in case that ACCMT was updated from < 1.1.0 or 1.2.3 version to a higher one,
                # this fixes it.
                status_dict["evaluations_done"] = 0
                status_dict["additional_metrics"] = {}

            if "patience_left" not in status_dict:
                # in case that ACCMT was updated from < 1.7.7 to a higher one, this fixes it.
                status_dict["patience_left"] = self.patience
        else:
            status_dict = {
                "best_train_loss": float("inf"),
                "best_valid_loss": float("inf"),
                "epoch": 0,
                "epoch_step": 0,
                "global_step": 0,
                "eval_global_step": 0,
                "evaluations_done": 0,
                "train_track_loss": 0,
                "additional_metrics": {},
                "patience_left": self.patience,
            }
        module.status_dict = status_dict
        self.monitor._set_extra(self.accelerator, status_dict, self.train_loss_metric_name, self.val_loss_metric_name)

        if module._implemented_collate_fn_train:
            self.collate_fn_train = module.collate_fn_train

        if module._implemented_collate_fn_val:
            self.collate_fn_val = module.collate_fn_val

        is_tuple = hasattr(self.hps.batch_size, "__len__")
        module.batch_size = self.hps.batch_size
        train_batch_size = self.hps.batch_size[0] if is_tuple else self.hps.batch_size
        val_batch_size = self.hps.batch_size[1] if is_tuple and len(self.hps.batch_size) > 1 else self.hps.batch_size
        dl_args = {
            "pin_memory": self.dataloader_pin_memory,
            "num_workers": self.dataloader_num_workers,
            "drop_last": self.dataloader_drop_last,
        }

        train_dataloader = module.get_train_dataloader()
        if train_dataset is not None and train_dataloader is None:
            shuffle_train = self.shuffle_train if self.sampler is None else None
            samplers = None
            if isinstance(self.sampler, list):
                samplers = []
                for sampler in self.sampler:
                    if issubclass(sampler.__class__, BaseSampler):
                        samplers.append(sampler(self.accelerator))
                    else:
                        samplers.append(sampler)
            else:
                samplers = (
                    self.sampler(self.accelerator) if issubclass(self.sampler.__class__, BaseSampler) else self.sampler
                )
            train_dataloader = DataLoader(
                train_dataset,
                shuffle=shuffle_train,
                sampler=samplers,
                batch_size=train_batch_size,
                collate_fn=self.collate_fn_train,
                **dl_args,
            )

            collate_fn_val = self.collate_fn_val if not self.disable_collate_fn_on_evaluation else None

            val_dataloader = module.get_validation_dataloader()
            if val_dataset is not None and val_dataloader is None:
                val_dataloader = DataLoader(
                    val_dataset,
                    shuffle=self.shuffle_validation,
                    batch_size=val_batch_size,
                    collate_fn=collate_fn_val,
                    **dl_args,
                )

            # conditionals
            _EVALUATION_EVERY_N_STEPS = (
                all([val_dataloader is not None, hasattr(module, "validation_step")])
                and self.evaluate_every_n_steps is not None
            )
            _CHECKPOINT_EVERY_N_STEPS = self.enable_checkpointing and self.checkpoint_strat == "step"
            _CHECKPOINT_AFTER_EVALUATION = self.enable_checkpointing and self.checkpoint_strat == "eval"
            _CHECKPOINT_WHEN_EPOCH_ENDS = self.enable_checkpointing and self.checkpoint_strat in {"epoch", "eval"}

            if val_dataloader is None and self.model_saving == "best_valid_loss":
                self.model_saving = "best_train_loss"

            optimizer = module.get_optimizer()
            if optimizer is None:
                optimizer = self._get_optimizer(model)

            scheduler = module.get_scheduler(
                optimizer, round(len(train_dataloader) / self.accelerator.num_processes), self.hps.epochs
            )
            if self.hps.scheduler is not None and scheduler is None:
                scheduler = self._get_scheduler(
                    optimizer, -1, round(len(train_dataloader) / self.accelerator.num_processes), self.hps.epochs
                )
                # -1 for last_epoch since Accelerate will take care of recovering the progress

            if self.log_with is not None and DEBUG_MODE < 1:
                self._initialize_trackers()

            unwrapped_model = model
            unwrapped_teacher = teacher
            unwrapped_optimizer = optimizer
            unwrapped_train_dataloader = train_dataloader
            unwrapped_val_dataloader = val_dataloader
            unwrapped_scheduler = scheduler

            if self.accelerator.distributed_type == DistributedType.FSDP:
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
                module.model = model  # force module.model to be wrapped to not have problems with dimmensions

            # NOTE: we aren't forcing module.model to be wrapped for other distributed types since we haven't seen any
            # issues with training, and it's actually a little bit faster doing inference with the model directly on the Module class.

            self.model = model

            if scheduler is not None:
                self.accelerator.register_for_checkpointing(scheduler)

            if self.resume:
                self.callback.on_resume()
                if os.path.exists(self.checkpoint):
                    self.accelerator.load_state(f"{self.checkpoint}/{CHECKPOINT_PATH}")
                else:
                    raise FileNotFoundError(f"{self.checkpoint} was not found.")

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.module.state = TrainingState(
            model=unwrapped_model,
            wrapped_model=model,
            teacher=unwrapped_teacher,
            wrapped_teacher=teacher,
            optimizer=unwrapped_optimizer,
            wrapped_optimizer=optimizer,
            train_dataloader=unwrapped_train_dataloader,
            wrapped_train_dataloader=train_dataloader,
            val_dataloader=unwrapped_val_dataloader,
            wrapped_val_dataloader=val_dataloader,
            scheduler=unwrapped_scheduler,
            wrapped_scheduler=scheduler,
        )
        self.module.device = self.accelerator.device

        self.callback.module = module
        self.callback.trainer = self
        self.callback.state = self.module.state

        self.callback.on_fit_start()

        if (
            self.eval_when_start
            and "evaluations_done" in status_dict
            and status_dict["evaluations_done"] == 0
            and DEBUG_MODE < 5
        ):
            self._eval(module, model, val_dataloader, status_dict, 0, self.hps.epochs, disable_train_loss=True)

        NUM_TRAIN_STEPS = len(self.train_dataloader)
        cleanup()
        first_epoch = True
        epoch = None
        try:
            for epoch in range(status_dict["epoch"], self.hps.epochs):
                status_dict["epoch"] = epoch
                self.monitor.log_epoch()
                initial_step = 0
                set_seed(epoch)
                if self.shuffle_train:
                    train_dataloader.set_epoch(epoch)
                if first_epoch and "skip_batches" in status_dict:
                    _train_dataloader = self.accelerator.skip_first_batches(
                        train_dataloader, status_dict["skip_batches"]
                    )
                    initial_step = status_dict["skip_batches"]
                else:
                    _train_dataloader = train_dataloader
                cleanup()
                model.train()
                self.callback.on_epoch_start()
                for step, batch in tqdm(
                    iterable=enumerate(_train_dataloader, initial_step),
                    total=len(train_dataloader),
                    initial=initial_step,
                    desc=f"Epoch {epoch}/{self.hps.epochs}",
                    unit="batch",
                ):
                    status_dict["epoch_step"] = step
                    CHECKPOINT_EVERY_N_STEPS = (
                        _CHECKPOINT_EVERY_N_STEPS and (status_dict["global_step"] + 1) % self.checkpoint_every == 0
                    )
                    EVALUATION_EVERY_N_STEPS = (
                        _EVALUATION_EVERY_N_STEPS
                        and (status_dict["global_step"] + 1) % self.evaluate_every_n_steps == 0
                    )
                    if status_dict["global_step"] % self.log_every == 0:
                        status_dict["learning_rate"] = (
                            scheduler.get_last_lr()[-1] if scheduler is not None else optimizer.param_groups[0]["lr"]
                        )
                        self.monitor.log_learning_rate()
                        self.monitor.log_cpu_utilization()
                        self.monitor.log_gpu_utilization()

                    self._train_logic(module, optimizer, batch, scheduler, status_dict)

                    if (
                        self.cleanup_cache_every_n_steps is not None
                        and status_dict["global_step"] % self.cleanup_cache_every_n_steps == 0
                    ):
                        cleanup()

                    if CHECKPOINT_EVERY_N_STEPS and DEBUG_MODE < 3:
                        self._save_checkpoint(
                            epoch, status_dict["epoch_step"] + 1, status_dict, status_dict["epoch_step"] + 1
                        )

                    if EVALUATION_EVERY_N_STEPS and DEBUG_MODE < 5:
                        self._eval(module, model, val_dataloader, status_dict, epoch, self.hps.epochs)
                        CHECKPOINT_AFTER_EVALUATION = (
                            _CHECKPOINT_AFTER_EVALUATION
                            and (status_dict["evaluations_done"] + 1) % self.checkpoint_every == 0
                        )
                        if CHECKPOINT_AFTER_EVALUATION and DEBUG_MODE < 3:
                            self._save_checkpoint(
                                epoch, status_dict["epoch_step"] + 1, status_dict, status_dict["epoch_step"] + 1
                            )
                    self.accelerator.wait_for_everyone()

                CHECKPOINT_WHEN_EPOCH_ENDS = (
                    _CHECKPOINT_WHEN_EPOCH_ENDS and (epoch + 1) % self.checkpoint_every == 0
                ) or (
                    _CHECKPOINT_AFTER_EVALUATION and (status_dict["evaluations_done"] + 1) % self.checkpoint_every == 0
                )

                if self.report_train_loss_per_epoch:
                    loss_report = self.train_track_loss / NUM_TRAIN_STEPS
                    loss_report = self.accelerator.reduce(loss_report, reduction="mean") * self.grad_accumulation_steps
                    status_dict["train_loss"] = loss_report.item()
                    self.monitor.log_train_loss(epoch=status_dict["epoch"])
                    # reset track loss
                    self.train_track_loss.zero_()

                if self.evaluate_every_n_steps is None and DEBUG_MODE < 5:
                    self._eval(module, model, val_dataloader, status_dict, epoch, self.hps.epochs)

                if CHECKPOINT_WHEN_EPOCH_ENDS and DEBUG_MODE < 3:
                    self._save_checkpoint(epoch + 1, 0, status_dict, 0)

                first_epoch = False
                cleanup()

            if (
                self.eval_when_finish
                and self.evaluate_every_n_steps is not None
                and epoch is not None
                and DEBUG_MODE < 5
            ):
                self._eval(module, model, val_dataloader, status_dict, epoch, self.hps.epochs, disable_train_loss=True)

            self.callback.on_epoch_end()
        except RuntimeError as e:
            self.callback.on_runtime_error(e)
            exception_str = str(e).lower()
            if "out of memory" in exception_str:
                self.callback.on_cuda_out_of_memory(e)

            if (
                "out of memory" in exception_str
                and any(handler in self.handlers for handler in [Handler.CUDA_OUT_OF_MEMORY, Handler.ALL])
                and DEBUG_MODE < 3
            ):
                self.accelerator.print(time_prefix(), "Forcing checkpointing due to CudaOutOfMemory error.")
                self._save_checkpoint(epoch, status_dict["epoch_step"], status_dict, status_dict["epoch_step"])
            elif any(handler in self.handlers for handler in [Handler.ANY, Handler.ALL]) and DEBUG_MODE < 3:
                self.accelerator.print(time_prefix(), "Forcing checkpointing due to a RunTime error.")
                self._save_checkpoint(epoch, status_dict["epoch_step"], status_dict, status_dict["epoch_step"])
            else:
                self.accelerator.print(e)
                traceback.print_exc()
        except KeyboardInterrupt as e:
            self.callback.on_runtime_error(e)
            if any(handler in self.handlers for handler in [Handler.KEYBOARD, Handler.ALL]) and DEBUG_MODE < 3:
                self.accelerator.print(time_prefix(), "Forcing checkpointing due to manual keyboard interrupt.")
                self._save_checkpoint(epoch, status_dict["epoch_step"], status_dict, status_dict["epoch_step"])
            else:
                self.accelerator.print(time_prefix(), "Manual keyboard interrupt.")
                traceback.print_exc()
        except Exception as e:
            self.callback.on_exception(e)
            if any(handler in self.handlers for handler in [Handler.ANY, Handler.ALL]) and DEBUG_MODE < 3:
                self.accelerator.print(time_prefix(), "Forcing checkpointing due to an exception.")
                self._save_checkpoint(epoch, status_dict["epoch_step"], status_dict, status_dict["epoch_step"])
            else:
                self.accelerator.print(e)
                traceback.print_exc()

        if epoch is None:
            raise RuntimeError(
                "Apparently you are trying to resume a training process that has already been finished."
            )
        self.callback.on_fit_end()
        self.accelerator.end_training()

    @torch.inference_mode()
    def _eval(
        self,
        module,
        model,
        val_dataloader,
        status_dict,
        epoch,
        epochs,
        disable_train_loss=False,
        disable_val_loss=False,
    ):
        cleanup()
        model.eval()

        if val_dataloader is not None:
            self.callback.on_evaluation_start()
            set_seed(epoch)
            if self.shuffle_validation:
                val_dataloader.set_epoch(epoch)
            for batch in tqdm(
                iterable=val_dataloader,
                total=len(val_dataloader),
                desc=f"Evaluating in Epoch {epoch}/{epochs}",
                unit="batch",
            ):
                self._validation_logic(module, batch, status_dict)

            # only rank 0 will be in charge of calculating metrics to avoid system overhead
            if self.accelerator.is_main_process:
                for metric in self.metrics:
                    metric_dict = metric._compute()

                    for m, v in metric_dict.items():
                        if not isinstance(v, (float, int)):
                            continue
                        status_dict["additional_metrics"][m] = v

            self._log_val_loss(status_dict, total=len(val_dataloader))
            self.monitor.log_additional_metrics()
            status_dict["evaluations_done"] += 1

        self.callback.on_evaluation_end()

        model.train()
        cleanup()

        if self.model_saving is not None and DEBUG_MODE < 3:
            self._save_model_on_criteria(model, status_dict, disable_train_loss, disable_val_loss)

    def _train_logic(self, module, optimizer, batch, scheduler, status_dict):
        self.callback.on_before_training_step(batch)
        loss = module.training_step(batch)
        self.callback.on_after_training_step()
        should_step = (status_dict["epoch_step"] + 1) % self.grad_accumulation_steps == 0
        loss = loss / self.grad_accumulation_steps

        with torch.inference_mode():
            detached_loss = loss.detach()
            self.train_total_loss += detached_loss
            self.train_track_loss += detached_loss
            status_dict["train_track_loss"] = self.train_track_loss

        log_every = self.log_every * self.grad_accumulation_steps
        if not self.report_train_loss_per_epoch and status_dict["global_step"] % log_every == 0:
            loss_report = self.train_track_loss / log_every
            loss_report = self.accelerator.reduce(loss_report, reduction="mean") * self.grad_accumulation_steps
            status_dict["train_loss"] = loss_report.item()
            self.monitor.log_train_loss()
            self.train_track_loss.zero_()  # reset track loss

        if self.accelerator.distributed_type == DistributedType.MULTI_GPU:
            self.model.require_backward_grad_sync = should_step  # for gradient sync when using DDP

        if not self.module._extended:
            self.callback.on_before_backward(loss)
            self.accelerator.backward(loss)
            self.callback.on_after_backward()

        if self.accelerator.sync_gradients and self.accelerator.is_main_process:
            norm = None
            if self.clip_grad is not None:
                norm = self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad)
            elif self.monitor.grad_norm:
                norm = self._get_grad_norm()

            if norm is not None and self.monitor.grad_norm:
                status_dict["grad_norm"] = norm
                self.monitor.log_grad_norm()

        if should_step and not self.module._extended:
            self.callback.on_before_optimizer_step(optimizer)
            optimizer.step()
            self.callback.on_after_optimizer_step(optimizer)
            if scheduler is not None:
                self.callback.on_before_scheduler_step(scheduler)
                scheduler.step()
                self.callback.on_after_scheduler_step(scheduler)

            self.callback.on_before_zero_grad(optimizer)
            optimizer.zero_grad(set_to_none=self.set_to_none)
            self.callback.on_after_zero_grad(optimizer)

        status_dict["global_step"] += 1

    def _track_val_loss(self, loss):
        detached_loss = loss.detach()
        self.val_total_loss += detached_loss
        self.val_track_loss += detached_loss

    def _log_val_loss(self, status_dict, total):
        loss_report = self.val_track_loss / total
        loss_report = self.accelerator.reduce(loss_report, reduction="mean")
        status_dict["validation_loss"] = loss_report.item()
        self.monitor.log_validation_loss()
        self.val_track_loss.zero_()  # reset val loss

    def _validation_logic(self, module, batch, status_dict):
        self.callback.on_before_validation_step(batch)
        metrics_dict = module.validation_step(batch)
        self.callback.on_after_validation_step()

        self._track_val_loss(metrics_dict["loss"])

        for metric in self.metrics:
            metric_compute_arguments = metrics_dict[metric.name]
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

            if self.accelerator.is_main_process and metric_compute_arguments[0] is not None:
                metric.add_batch(*metric_compute_arguments)

        status_dict["eval_global_step"] += 1

    def _save_model(self, model, status_dict, wait_for_everyone=True, model_path=None):
        self.accelerator.print(time_prefix(), "Saving model...")

        model_path = self.model_path if model_path is None else model_path
        os.makedirs(model_path, exist_ok=True)  # re-create folder in case it was deleted

        PATH_DOES_NOT_EXIST = not os.path.exists(self.model_path) and self.accelerator.is_main_process
        if PATH_DOES_NOT_EXIST and not wait_for_everyone:
            os.makedirs(model_path, exist_ok=True)
        if wait_for_everyone:
            if PATH_DOES_NOT_EXIST:
                os.makedirs(model_path, exist_ok=True)
            self.accelerator.wait_for_everyone()

        unwrapped_model = self.accelerator.unwrap_model(model)
        state_dict = unwrapped_model.state_dict() if not self.compile else unwrapped_model._orig_mod.state_dict()
        if hasattr(unwrapped_model, "save_pretrained"):
            unwrapped_model.save_pretrained(
                model_path,
                is_main_process=self.accelerator.is_main_process,
                state_dict=state_dict,
                max_shard_size=self.max_shard_size,
                save_function=self.accelerator.save,
                safe_serialization=self.safe_serialization,
            )
        else:
            self.accelerator.save(
                state_dict, f"{model_path}/pytorch_model.pt", safe_serialization=self.safe_serialization
            )

        if self.accelerator.is_main_process:
            save_status(status_dict, to=f"{model_path}/{STATUS_PATH}")

        self.accelerator.print(time_prefix(), "Model saved.")

    def _save_model_on_criteria(self, model, status_dict, disable_train_loss=False, disable_val_loss=False):
        # use 'disable_train_loss' when evaluating at the beginning
        if self.model_saving is None:
            return

        self.accelerator.wait_for_everyone()

        if status_dict["patience_left"] > 0:
            status_dict["patience_left"] -= 1

        if status_dict["patience_left"] == 0:
            self.accelerator.print(time_prefix(), "Ran out of patience. Finishing process...")
            self.accelerator.end_training()

            self.accelerator.wait_for_everyone()
            exit(0)

        train_length = (
            self.evaluate_every_n_steps if self.evaluate_every_n_steps is not None else len(self.train_dataloader)
        )

        avg_valid_loss = self.val_total_loss / (len(self.val_dataloader) if self.val_dataloader is not None else -1)
        avg_valid_loss = self.accelerator.reduce(avg_valid_loss, reduction="mean")
        avg_train_loss = self.train_total_loss / train_length
        avg_train_loss = self.accelerator.reduce(avg_train_loss, reduction="mean")

        if self.accelerator.is_main_process:
            avg_valid_loss = avg_valid_loss.item()
            avg_train_loss = avg_train_loss.item()

            saving_criteria = {}
            for metric in self.metrics:
                best_metric_str = f"best_{metric.main_metric}"
                if best_metric_str not in status_dict:
                    status_dict[best_metric_str] = -1

                prev = status_dict[best_metric_str]
                new = status_dict["additional_metrics"][metric.main_metric]
                compare = operator_map[metric.comparator]
                is_better = compare(new, prev)
                best = new if is_better else prev

                saving_criteria[best_metric_str] = (
                    is_better and new < self.model_saving_below and new > self.model_saving_above
                )
                status_dict[best_metric_str] = best

            if not disable_val_loss:
                saving_criteria["best_valid_loss"] = (
                    avg_valid_loss < status_dict["best_valid_loss"]
                    and avg_valid_loss < self.model_saving_below
                    and avg_valid_loss > self.model_saving_above
                )
                status_dict["best_valid_loss"] = (
                    avg_valid_loss
                    if avg_valid_loss < status_dict["best_valid_loss"]
                    else status_dict["best_valid_loss"]
                )

            if not disable_train_loss:
                saving_criteria["best_train_loss"] = (
                    avg_train_loss < status_dict["best_train_loss"]
                    and avg_train_loss < self.model_saving_below
                    and avg_train_loss > self.model_saving_above
                )
                status_dict["best_train_loss"] = (
                    avg_train_loss
                    if avg_train_loss < status_dict["best_train_loss"]
                    else status_dict["best_train_loss"]
                )

            saving_criteria["always"] = True

            for model_saving in self.model_saving:
                if saving_criteria[model_saving]:
                    model_path = f"{self.model_path}/{model_saving}"
                    self._save_model(model, status_dict, wait_for_everyone=False, model_path=model_path)

        self.val_total_loss.zero_()  # reset val total loss
        self.train_total_loss.zero_()  # reset train total loss

    def _save_checkpoint(self, epoch, epoch_step, status_dict, skip_batches):
        self.callback.on_save_checkpoint()
        self.accelerator.print(time_prefix(), "Saving checkpoint...")
        if self.accelerator.is_main_process:
            if not os.path.exists(self.checkpoint):
                os.makedirs(self.checkpoint, exist_ok=True)
            if not os.path.exists(f"{self.checkpoint}/{CHECKPOINT_PATH}"):
                os.makedirs(f"{self.checkpoint}/{CHECKPOINT_PATH}", exist_ok=True)
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(f"{self.checkpoint}/{CHECKPOINT_PATH}", safe_serialization=self.safe_serialization)
        if self.accelerator.is_main_process:
            status = status_dict.copy()
            status["epoch"] = epoch
            status["epoch_step"] = epoch_step
            if self.checkpoint_strat == "step" or (
                self.checkpoint_strat == "eval" and self.evaluate_every_n_steps is not None
            ):
                status["skip_batches"] = skip_batches
            save_status(status, to=f"{self.checkpoint}/{STATUS_PATH}")
        self.accelerator.print(time_prefix(), "Checkpoint saved.")

    def _get_optimizer(self, model):
        optimizer = self.hps.optim
        fused_available = "fused" in inspect.signature(optimizer).parameters
        optim_kwargs = self.hps.optim_kwargs
        optim_kwargs["fused"] = fused_available and "cuda" in self.accelerator.device.type

        filtered_kwargs = self._filter_kwargs(optim_kwargs, optimizer)

        return optimizer(model.parameters(), **filtered_kwargs)

    def _filter_kwargs(self, _kwargs: dict, fn):
        try:
            return {k: v for k, v in _kwargs.items() if k in fn.__init__.__code__.co_varnames}
        except AttributeError:
            signature = inspect.signature(fn)
            parameters = list(signature.parameters.keys())
            return {k: v for k, v in _kwargs.items() if k in parameters}

    def _get_scheduler(self, optimizer, last_epoch, steps_per_epoch, epochs):
        steps_per_epoch = round(steps_per_epoch / self.grad_accumulation_steps)
        schlr_kwargs = self.hps.scheduler_kwargs
        schlr_kwargs["last_epoch"] = last_epoch
        schlr_kwargs["steps_per_epoch"] = steps_per_epoch
        total_steps = steps_per_epoch * epochs
        schlr_kwargs["num_training_steps"] = total_steps
        schlr_kwargs["epochs"] = epochs
        if "num_warmup_steps" in schlr_kwargs and isinstance(schlr_kwargs["num_warmup_steps"], float):
            if schlr_kwargs["num_warmup_steps"] < 0.0 or schlr_kwargs["num_warmup_steps"] > 1.0:
                raise ValueError(
                    "If 'num_warmup_steps' is a ratio (float value), it needs to be a value between 0 and 1."
                )
            schlr_kwargs["num_warmup_steps"] = round(total_steps * schlr_kwargs["num_warmup_steps"])
        elif "warmup_ratio" in schlr_kwargs:
            if schlr_kwargs["warmup_ratio"] > 1.0:
                raise ValueError(
                    "'warmup_ratio' value in scheduler configuration needs to be a value between 0 and 1."
                )
            schlr_kwargs["num_warmup_steps"] = round(total_steps * schlr_kwargs["warmup_ratio"])

        scheduler = self.hps.scheduler
        filtered_kwargs = self._filter_kwargs(schlr_kwargs, scheduler)

        return scheduler(optimizer, **filtered_kwargs)

    def _initialize_trackers(self):
        if self.accelerator.is_main_process:
            for logger in self.log_with:
                if logger.tracker == LoggerType.MLFLOW and is_url(self.logging_dir):
                    import mlflow

                    mlflow.set_tracking_uri(self.logging_dir)
                    break

    def _get_collate_fn_pipeline(self, new_collate_fn):
        def collate_fns(batch):
            for collate_fn in new_collate_fn:
                batch = collate_fn(batch)

            return batch

        return collate_fns

    def _get_grad_norm(self) -> float:
        return math.sqrt(
            sum([torch.norm(p.grad.detach()) ** 2 for p in self.model.parameters() if p.grad is not None]).item()
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
        if self.accelerator.is_main_process and DEBUG_MODE < 1:
            import mlflow

            mlflow.log_artifact(path, **kwargs)
