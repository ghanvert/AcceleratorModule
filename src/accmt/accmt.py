import inspect
import math
import numpy as np
import torch

def allow_tf32(flag=True):
    """Enable or disable the use of TensorFloat32."""
    torch.set_float32_matmul_precision("high" if flag else "highest")

allow_tf32()

from abc import ABC
from accelerate import Accelerator, DataLoaderConfiguration, DistributedType
from accelerate.utils import ProjectConfiguration, InitProcessGroupKwargs, LoggerType, tqdm, set_seed
from evaluate import load
from .handlers import Handler
import os
import traceback
import torch
import torch.nn as nn
import gc
from .utils import get_number_and_unit, is_url, time_prefix, combine_dicts, save_status, read_status, suppress_print_and_warnings
from .dataloader_samplers import BaseSampler
from .monitor import Monitor
from torch.utils.data import Dataset, DataLoader
from typing_extensions import Any, Optional, Union, override
from .hyperparameters import HyperParameters
from .metrics import MetricComparator
from torch.distributed import destroy_process_group
from datetime import timedelta

CHECKPOINT_PATH = "checkpoint"
STATUS_PATH = "status.json"

init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=86400))
dataloader_config = DataLoaderConfiguration(use_seedable_sampler=True)
accelerator = Accelerator(kwargs_handlers=[init_kwargs], dataloader_config=dataloader_config, step_scheduler_with_optimizer=False)

class AcceleratorModule(ABC):
    """
    Super class to define training and validation logic without the need
    to write a training loop.

    The constructor of this class must implement `self.model`, specifying the model
    from `torch.nn.Module`. `self.teacher` is also a reserved property for teacher-student 
    approaches.

    Methods:
        `forward` (*optional*):
            Defines the flow of data of model. If not implemented, `__call__`
            will not be possible (e.g. `self(...)`). Should return the model output.
        `training_step` (*optional*):
            Defines the training logic. Must return a loss `torch.Tensor` (scalar).
        `validation_step` (*optional*):
            Defines the validation logic. Must return a loss `torch.Tensor` (scalar).
            If not implemented, no validation will be executed.
        `step` (*optional*):
            Defines the training and validation logics. This is useful when training
            and validation logics are the same. Must return a loss `torch.Tensor` (scalar).

            NOTE: Cannot define `step` together with `training_step` and/or 
            `validation_step`.
        `collate_fn` (*optional*):
            Defines the collator function for DataLoader.
        `get_optimizer` (*optional*):
            Defines the optimizer. Must return the optimizer itself.
        `get_scheduler` (*optional*):
            Defines the scheduler. Must return the scheduler itself.
        `get_train_dataloader` (*optional*):
            Defines the train DataLoader. Must return a torch `DataLoader`.
        `get_validation_dataloader` (*optional*):
            Defines the validation DataLoader. Must return a torch `DataLoader`.
    
    Special methods (no implementation required):
        `__call__`:
            When calling this module, it will execute `forward` method.
        `__repr__`:
            When reproducing this module (e.g. Jupyter Notebook cell), this will print
            the model structure from `torch.nn.Module` specified in `self.model`.
        `__str__`:
            When printing this module or using it as a `str` type, this will represent
            the `torch.nn.Module` specified in `self.model`.
        `__len__`:
            When casting this module with `len` Python function, it will return the
            number of parameters of the base model specified in `self.model` from
            `torch.nn.Module`.
    """
    _implemented_collate_fn = False
    _accelerator = accelerator
    _log_every: int = 1
    device = _accelerator.device
    status_dict: dict = None

    @override
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Defines the flow of data."""

    @override
    def step(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Defines the logic for both training and validation."""
    
    @override
    def training_step(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Defines the training logic. Must return a loss tensor (scalar)."""
    
    @override
    def validation_step(self, *args: Any, **kwargs: Any) -> Union[torch.Tensor, dict]:
        """
        Defines the validation logic. Must return a loss tensor (scalar) or a dictionary containing 
        each metric with predictions and targets, and also the loss value in the dictionary 
        (in case you are calculating additional metrics).

        Example:
            ```
            # format is ==> "metric": (predictions, targets)
            return {
                "loss": validation_loss_tensor (scalar tensor).
                "accuracy": (accuracy_predictions, accuracy_targets),
                "bleu": (bleu_predictions, bleu_targets)
            }
            ```

        NOTE: Return type must be:
            - `torch.Tensor` when `Trainer` does not use `additional_metrics`.
            - `dict` when `Trainer` uses `additional_metrics`.
        """
    
    @override
    def test_step(self, *args: Any, **kwargs: Any) -> dict:
        """
        Defines the test logic. Must return a dictionary containing each metric with predictions and targets.
        
        Example:
            ```
            # format is ==> "metric": (predictions, targets)
            return {
                "accuracy": (accuracy_predictions, accuracy_targets),
                "bleu": (bleu_predictions, bleu_targets)
            }
            ```
        """

    @override
    def collate_fn(self, batch: list) -> Any:
        """Defines a collate function for PyTorch DataLoader."""

    @override
    def get_optimizer(self, *args: Any, **kwargs: Any) -> Any:
        """Defines a custom PyTorch optimizer logic here."""

    @override
    def get_scheduler(self, optimizer: Any, steps_per_epoch: int, epochs: int) -> Any:
        """Defines a custom PyTorch scheduler logic here."""

    @override
    def get_train_dataloader(self, *args: Any, **kwargs: Any) -> Any:
        """Defines a custom PyTorch DataLoader class for training."""

    @override
    def get_validation_dataloader(self, *args: Any, **kwargs: Any) -> Any:
        """Defines a custom PyTorch DataLoader class for validation."""

    @override
    def get_test_dataloader(self, *args: Any, **kwargs: Any) -> Any:
        """Defines a custom PyTorch DataLoader class for test."""

    @accelerator.on_main_process
    def log(self, values: dict, log_kwargs: dict | None = {}):
        train_or_eval = "global_step" if self.model.training else "eval_global_step"
        if (self.status_dict[train_or_eval]+1) % self._log_every == 0:
            accelerator.log(values, step=self.status_dict[train_or_eval], log_kwargs=log_kwargs)
    
    def __init_subclass__(cls, **kwargs):
        if (
            cls.training_step == AcceleratorModule.training_step and
            cls.validation_step == AcceleratorModule.validation_step and
            cls.step == AcceleratorModule.step
        ):
            raise TypeError(
                "Subclasses of 'Trainer' must override 'training_step' and/or "
                "'validation_step' methods. If you want training and validation "
                "logics to be the same, then override 'step' method."
            )
        elif (
            (cls.training_step != AcceleratorModule.training_step or
            cls.validation_step != AcceleratorModule.validation_step)
            and cls.step != AcceleratorModule.step
        ):
            raise TypeError(
                "Subclasses of 'Trainer' cannot have training or validation logic "
                "together with 'step' method. It is either 'step', or at least of "
                "'training_step' or 'validation_step' methods."
            )
        
        if cls.collate_fn != AcceleratorModule.collate_fn:
            cls._implemented_collate_fn = True

        super().__init_subclass__(**kwargs)

    def __call__(self, *args: Any, **kwargs: Any):
        return self.forward(*args, **kwargs)

    def __repr__(self):
        return self.model
    
    def __str__(self):
        return self.model.__repr__()
    
    def __len__(self):
        return sum(p.numel() for p in self.model.parameters())
    
    @classmethod
    def from_hf(cls, path: str, type: Union[str, Any] = None, **kwargs: Optional[Any]):
        """
        Build a custom AcceleratorModule for HuggingFace's transformers library. It simply replaces the following standard:
        
        ```
        class Module(AcceleratorModule):
            def __init__(self):
                self.model = AutoModel.from_pretrained(path, **kwargs)

            def step(self, batch):
                return self.model(**batch).loss
        ```

        Args:
            path (`str`):
                Path for HuggingFace model.
            type (`str` or `Any`):
                Model type in transformers library. It can be the class itself or a string (no need for imports).
            kwargs (`Any`):
                Keyword arguments for `from_pretrained` function for model initialization.
        """
        if isinstance(type, str):
            import importlib
            module = importlib.import_module("transformers")
            type = getattr(module, type)
        elif type is None:
            from transformers import AutoModel
            type = AutoModel
        
        class Module(AcceleratorModule):
            def __init__(self):
                self.model = type.from_pretrained(path, **kwargs)

            def step(self, batch):
                return self.model(**batch).loss
            
        return Module()


class Trainer:
    """
    Class to implement the training configuration.
    """
    @classmethod
    def from_config(cls,
                    config: Union[str, dict],
                    log_with: Optional[Union[Any, list]] = None,
                    sampler: Optional[Union[Any, list]] = None,
                    collate_fn: Optional[Any] = None
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
            collate_fn (`function` or `list`, *optional*, defaults to `None`):
                Collate function to be implemented in dataloaders.
        """
        assert isinstance(config, (str, dict)), "'config' needs to be either a path to a file, or a dictionary."
        if isinstance(config, str):
            import yaml
            config = yaml.safe_load(open(config))

        return Trainer(**config, log_with=log_with, sampler=sampler, collate_fn=collate_fn)

    def __init__(self,
                hps_config: Union[str, dict, HyperParameters],
                model_path: str,
                track_name: Optional[str] = None,
                checkpoint: Optional[str] = None,
                resume: Optional[bool] = None,
                model_saving: Optional[str] = "best_valid_loss",
                evaluate_every_n_steps: Optional[int] = None,
                checkpoint_every: Optional[str] = "epoch",
                logging_dir: str = "logs",
                log_with: Optional[Union[Any, list]] = None,
                log_every: int = 1,
                grad_accumulation_steps: Optional[int] = None,
                clip_grad: Optional[float] = None,
                set_to_none: bool = True,
                shuffle_train: bool = True,
                shuffle_validation: bool = False,
                shuffle_test: bool = False,
                sampler: Optional[Union[Any, list]] = None,
                model_saving_below: Optional[float] = None,
                model_saving_above: Optional[float] = None,
                collate_fn: Optional[Any] = None,
                max_shard_size: str = "10GB",
                safe_serialization: bool = False,
                compile: bool = False,
                scale_learning_rate: bool = False,
                train_loss_metric_name: str = "train_loss",
                val_loss_metric_name: str = "val_loss",
                dataloader_pin_memory: bool = True,
                dataloader_num_workers: int = 0,
                report_loss_after_eval: bool = True,
                handlers: Optional[Union[list, Any]] = None,
                eval_when_finish: bool = True,
                eval_when_start: bool = False,
                verbose: bool = False,
                monitor: Optional[Monitor] = None,
                additional_metrics: Optional[Union[list[str], str]] = None,
                metrics_num_process: int = 1,
                **kwargs: Optional[Any]
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
            model_saving (`str`, *optional*, defaults to `best_valid_loss`):
                Type of model saving. It can be one of the following values:

                - `"best_valid_loss"`: Saves the model whenever the validation loss is the best recorded.
                - `"best_train_loss"`: Saves the model whenever the training loss is the best recorded.
                - `"always"`: Saves the model always at the end of every evaluation.
                - in format of `"best_{METRIC}"`, where METRIC corresponds to the additional metric in `additional_metrics`.

                If not specified (`None`), model saving will be disabled.
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
                modified to be `log_every` * `grad_accumulation_steps`.
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
            shuffle_test (`bool`, *optional*, defaults to `False`):
                Whether to shuffle test DataLoader.
            sampler (`list` or `Any`, *optional*, defaults to `None`):
                Sampler (or list of samplers) for train DataLoader.
            model_saving_below (`float`, *optional*, defaults to `None`):
                Start saving model below this metric (based on `model_saving`).
            model_saving_above (`float`, *optional*, defaults to `None`):
                Start saving model above this metric (based on `model_saving`).
            collate_fn (`function` or `list`, *optional*, defaults to `None`):
                Collate function to be implemented in dataloaders. If `module` overrides `collate_fn` from
                `AcceleratorModule` class, then that function will be used instead of the one specified on
                this constructor. If a list of collate functions is given, then the every collate function will affect
                the batch in the given order.
            max_shard_size (`str`, *optional*, defaults to `10GB`):
                Max model shard size to be used.
            safe_serializartion (`bool`, *optional*, defaults to `False`):
                Whether to save model using safe tensors or the traditional PyTorch way. If `True`, some tensors
                will be lost.
            compile (`bool`, *optional*, defaults to `False`):
                Whether to call `torch.compile` on model (and teacher, if implemented).
            scale_learning_rate (`bool`, *optional*, defaults to `False`):
                Multiply learning rate by the number of processes as suggested in https://arxiv.org/pdf/1706.02677.
            train_loss_metric_name (`str`, *optional*, defaults to `train_loss`):
                Metric name for train loss in logs.
            val_loss_metric_name (`str`, *optional*, defaults to `val_loss`):
                Metric name for validation loss in logs.
            dataloader_pin_memory (`bool`, *optional*, defaults to `True`):
                Enables pin memory option in DataLoader.
            dataloader_num_workers (`int`, *optional*, defaults to `0`):
                Number of processes for DataLoader.
            report_loss_after_eval (`bool`, *optional*, defaults to `True`):
                Whether to report average validation loss after evaluation. If set to `False`, loss will be reported by every batch.
            handlers (`Any` or `list`, *optional*, defaults to `None`):
                Handler or List of handlers to catch errors and make a safe checkpoint.
            eval_when_finish (`bool`, *optional*, defaults to `True`):
                At the end of training, evaluate model on validation dataset (if available). This option is only valid when 
                `evaluate_every_n_steps` is not `None`.
            eval_when_start (`bool`, *optional*, defaults to `False`):
                Start training with evaluation (if available).
            verbose (`bool`, *optional*, defaults to `False`):
                NOTE: This is a preliminary feature, and it may not disable prints in some accelerate backends.

                Enable or disable prints from Accelerate backend (e.g. training initialization, specific checkpoints, warnings, etc). 
                You might want to enable this when debugging.
            monitor (`Monitor` or `dict`, *optional*, defaults to `None`):
                Monitor arguments to keep track of variables during training. If not specified, 'train_loss' and 'validation_loss' will 
                be set to `True` by default.

                NOTE: Learning rate, GPU and CPU monitoring will only be reported during training, not evaluation. Also, GPU and CPU 
                monitoring will only be reported on main process (index 0).
            additional_metrics (`list` or `str`, *optional*, defaults to `None`):
                Additional metrics to calculate on test (if given) or validation dataset. If none of these are given, no additional metrics 
                will be calculed.

                Available metrics are: accuracy, bertscore, bleu, bleurt, brier_score, cer, character, charcut_mt, chrf, f1, glue, precision, 
                r_squared, recall, mse, mean_iou.
            metrics_num_process (`int`, *optional*, defaults to `1`):
                Number of processes for metrics calculation. WARNING: when testing this feature, we noticed an internal bug when using 
                `metrics_num_process` > 1.
            kwargs (`Any`, *optional*):
                Extra arguments for specific `init` function in Tracker, e.g. `run_name`, `tags`, etc.
        """
        self.verbose = verbose
        assert isinstance(hps_config, (str, dict, HyperParameters)), f"'hps_file_config' needs to be either a string, dictionary or HyperParameters class."
        self.hps = HyperParameters.from_config(hps_config) if isinstance(hps_config, (str, dict)) else hps_config
        self.track_name = track_name
        self.checkpoint = checkpoint if checkpoint is not None else f"checkpoint-{model_path.split('/')[-1]}"
        self.resume = resume if resume is not None else os.path.exists(self.checkpoint) and len(os.listdir(self.checkpoint)) > 0
        self.model_path = model_path
        self.model_saving = model_saving.lower()
        assert "best_" in self.model_saving, "Format for model_saving is: 'best_{METRIC}'."
        _metric = self.model_saving.removeprefix("best_")
        _available_metrics = [attr for attr in dir(MetricComparator) if not attr.startswith("__")]
        assert self.model_saving in {"best_valid_loss", "best_train_loss", "always"} or _metric in _available_metrics, f"{self.model_saving} is not valid."
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
        if clip_grad is not None and accelerator.distributed_type == DistributedType.DEEPSPEED:
            accelerator.print(time_prefix(), "[WARNING] Clipping gradient using Trainer is not supported when running with DeepSpeed. Setting it to None.")
            clip_grad = None
        self.clip_grad = clip_grad
        self.set_to_none = set_to_none
        self.shuffle_train = shuffle_train
        self.shuffle_validation = shuffle_validation
        self.shuffle_test = shuffle_test
        self.sampler = sampler
        self.model_saving_below = model_saving_below if model_saving_below is not None else float("inf")
        self.model_saving_above = model_saving_above if model_saving_above is not None else float("-inf")
        self.collate_fn = self._get_collate_fn_pipeline() if isinstance(collate_fn, list) else collate_fn
        self.max_shard_size = max_shard_size
        self.safe_serialization = safe_serialization
        self.compile = compile
        self.scale_learning_rate = scale_learning_rate
        self.train_loss_metric_name = train_loss_metric_name
        self.val_loss_metric_name = val_loss_metric_name
        self.dataloader_pin_memory = dataloader_pin_memory
        self.dataloader_num_workers = dataloader_num_workers
        self.report_loss_after_eval = report_loss_after_eval
        self.handlers = handlers if isinstance(handlers, list) else [handlers]
        if self.handlers[0] is not None: raise NotImplementedError("'handlers' argument is not yet fully implemented.")
        self.eval_when_finish = eval_when_finish
        self.eval_when_start = eval_when_start
        self.monitor = monitor if isinstance(monitor, Monitor) else Monitor.from_config(monitor)
        self.monitor.grad_norm = self.monitor.grad_norm if accelerator.distributed_type == DistributedType.DEEPSPEED else False
        if self.monitor.grad_norm and accelerator.distributed_type == DistributedType.DEEPSPEED:
            accelerator.print(time_prefix(),
                              "[WARNING] Gradient norm monitoring is not yet supported when running with DeepSpeed. Setting it to False.")
            self.monitor.grad_norm = False
        if not self.monitor.val_equal_train and not report_loss_after_eval: self.monitor.val_equal_train = True
        self.additional_metrics = additional_metrics if isinstance(additional_metrics, list) or additional_metrics is None else [additional_metrics]

        if additional_metrics is not None:
            # pre-download metrics
            accelerator.wait_for_everyone()
            for additional_metric in self.additional_metrics:
                load(additional_metric) # dummy 
        
        if metrics_num_process > 1:
            accelerator.print(time_prefix(), "[WARNING] 'metrics_num_process' could not work")
        self.metrics_num_process = metrics_num_process
        self.init_kwargs = kwargs

        if isinstance(grad_accumulation_steps, int) and grad_accumulation_steps > 1:
            accelerator.gradient_accumulation_steps = grad_accumulation_steps
        accelerator.project_configuration = ProjectConfiguration(project_dir=".", logging_dir=logging_dir, total_limit=1)

        if log_with is not None:
            if not isinstance(log_with, list): log_with = [log_with]
            accelerator.log_with = [tracker.tracker for tracker in log_with]
            self.log_with = [tracker for tracker in log_with]

        # we need to calculate mean for these tensors.
        self.train_total_loss: torch.Tensor = None # loss tensor for evaluation
        self.train_track_loss: torch.Tensor = None # train loss tensor to be reported
        self.val_total_loss: torch.Tensor = None # val loss tensor for evaluation
        self.val_track_loss: torch.Tensor = None # val loss tensor to be reported

        self.train_dataloader: DataLoader = None
        self.val_dataloader: DataLoader = None
        self.test_dataloader: DataLoader = None

    def fit(self,
            module: Union[AcceleratorModule, str, Union[tuple[str, str], tuple[str, Any]]],
            train_dataset: Optional[Dataset] = None,
            val_dataset: Optional[Dataset] = None,
            test_dataset: Optional[Dataset] = None,
            **kwargs: Any
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
            test_dataset (`torch.utils.data.Dataset`, *optional*, defaults to `None`):
                `Dataset` class from PyTorch containing the test dataset logic. This dataset will be used to 
                calculate additional metrics like accuracy (specified in `additional_metrics` in Trainer).
            kwargs (`Any`):
                Keyword arguments for `from_pretrained` function for model initialization.
        """
        self.additional_metrics = self.additional_metrics if val_dataset is not None or test_dataset is not None else None
        self.monitor.additional_metrics = self.additional_metrics is not None and len(self.additional_metrics) > 0
        self.monitor._do_tracking = self.log_with is not None
        self.train_total_loss = torch.tensor(0.0, device=accelerator.device)
        self.train_track_loss = torch.tensor(0.0, device=accelerator.device)
        self.val_total_loss = torch.tensor(0.0, device=accelerator.device)
        self.val_track_loss = torch.tensor(0.0, device=accelerator.device)

        if test_dataset is not None and self.additional_metrics is None:
            raise ValueError("You must implement 'additional_metrics' when using test dataset.")

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
        with suppress_print_and_warnings(self.verbose):
            if torch.cuda.is_available():
                model.to(accelerator.device) # for optimizer to apply fused when available
                if teacher is not None:
                    teacher.to(accelerator.device)
            if self.compile:
                model = torch.compile(model)
                if teacher is not None:
                    teacher = torch.compile(teacher)

            if accelerator.distributed_type == DistributedType.FSDP:
                model = accelerator.prepare(model)

            os.makedirs(self.model_path, exist_ok=True)

            if self.resume:
                status_dict = read_status(f"{self.checkpoint}/{STATUS_PATH}")
                if "evaluations_done" not in status_dict:
                    # in case that ACCMT was updated from < 1.1.0 or 1.2.3 version to a higher one, 
                    # this fixes it.
                    status_dict["evaluations_done"] = 0
                    status_dict["additional_metrics"] = {}
                    status_dict["test_global_step"] = 0
            else:
                status_dict = {
                    "best_train_loss": float("inf"),
                    "best_valid_loss": float("inf"),
                    "epoch": 0,
                    "epoch_step": 0,
                    "global_step": 0,
                    "eval_global_step": 0,
                    "evaluations_done": 0,
                    "additional_metrics": {},
                    "test_global_step": 0
                }
            module.status_dict = status_dict
            self.monitor._set_extra(accelerator, status_dict, self.train_loss_metric_name, self.val_loss_metric_name)

            if module._implemented_collate_fn:
                self.collate_fn = module.collate_fn

            is_tuple = hasattr(self.hps.batch_size, "__len__")
            train_batch_size = self.hps.batch_size[0] if is_tuple else self.hps.batch_size
            val_batch_size = self.hps.batch_size[1] if is_tuple and len(self.hps.batch_size) == 2 else self.hps.batch_size
            test_batch_size = (
                self.hps.batch_size[2] if is_tuple and
                len(self.hps.batch_size) == 3 else (
                    self.hps.batch_size if not is_tuple else self.hps.batch_size[-1]
                )
            )
            dl_args = {
                "collate_fn": self.collate_fn,
                "pin_memory": self.dataloader_pin_memory,
                "num_workers": self.dataloader_num_workers
            }

            train_dataloader = module.get_train_dataloader()
            if train_dataset is not None and train_dataloader is None:
                shuffle_train = self.shuffle_train if self.sampler is None else None
                samplers = None
                if isinstance(self.sampler, list):
                    samplers = []
                    for sampler in self.sampler:
                        if issubclass(sampler.__class__, BaseSampler):
                            samplers.append(sampler(accelerator))
                        else:
                            samplers.append(sampler)
                else:
                    samplers = self.sampler(accelerator) if issubclass(self.sampler.__class__, BaseSampler) else self.sampler
                train_dataloader = DataLoader(train_dataset, shuffle=shuffle_train, sampler=samplers, batch_size=train_batch_size, **dl_args)

                val_dataloader = module.get_validation_dataloader()
                if val_dataset is not None and val_dataloader is None:
                    val_dataloader = DataLoader(val_dataset, shuffle=self.shuffle_validation, batch_size=val_batch_size, **dl_args)

                test_dataloader = module.get_test_dataloader()
                if test_dataset is not None and test_dataloader is None:
                    test_dataloader = DataLoader(test_dataset, shuffle=self.shuffle_test, batch_size=test_batch_size, **dl_args)
                
                # conditionals
                _EVALUATION_EVERY_N_STEPS = all([val_dataloader is not None, hasattr(module, "validation_step")]) and self.evaluate_every_n_steps is not None
                _CHECKPOINT_EVERY_N_STEPS = self.enable_checkpointing and self.checkpoint_strat == "step"
                _CHECKPOINT_AFTER_EVALUATION = self.enable_checkpointing and self.checkpoint_strat == "eval"
                _CHECKPOINT_WHEN_EPOCH_ENDS = self.enable_checkpointing and self.checkpoint_strat in {"epoch", "eval"}

                if val_dataloader is None and self.model_saving == "best_valid_loss":
                    self.model_saving = "best_train_loss"

                optimizer = module.get_optimizer()
                if optimizer is None:
                    optimizer = self._get_optimizer(model)

                if self.scale_learning_rate:
                    self._scale_learning_rate(optimizer)

                scheduler = module.get_scheduler(optimizer, round(len(train_dataloader)/accelerator.num_processes), self.hps.epochs)
                if self.hps.scheduler is not None and scheduler is None:
                    scheduler = self._get_scheduler(optimizer, -1, round(len(train_dataloader)/accelerator.num_processes), self.hps.epochs)
                    # -1 for last_epoch since Accelerate will take care of recovering the progress

                if self.log_with is not None:
                    self._initialize_trackers()

                if accelerator.distributed_type == DistributedType.FSDP:
                    train_dataloader, val_dataloader, test_dataloader, optimizer, scheduler, teacher = accelerator.prepare(
                        train_dataloader, val_dataloader, test_dataloader, optimizer, scheduler, teacher
                    )
                    module.model = model
                else:
                    model, train_dataloader, val_dataloader, test_dataloader, optimizer, scheduler, teacher = accelerator.prepare(
                        model, train_dataloader, val_dataloader, test_dataloader, optimizer, scheduler, teacher
                    )
                self.model = model

                if scheduler is not None:
                    accelerator.register_for_checkpointing(scheduler)

                if self.log_with is not None:
                    track_name = self.model_path.split("/")[-1] if self.track_name is None else self.track_name
                    init_kwargs = combine_dicts(*[tracker.init(**self.init_kwargs) for tracker in self.log_with])
                    accelerator.init_trackers(track_name, config=self.hps.get_config(), init_kwargs=init_kwargs)

                if self.resume:
                    if os.path.exists(self.checkpoint):
                        accelerator.load_state(f"{self.checkpoint}/{CHECKPOINT_PATH}")
                    else:
                        raise FileNotFoundError(f"{self.checkpoint} was not found.")

                if self.eval_when_start and "evaluations_done" in status_dict and status_dict["evaluations_done"] == 0:
                    self._eval(module, model, val_dataloader, test_dataloader, status_dict, 0, self.hps.epochs)

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        
        gc.collect()
        first_epoch = True
        try:
            for epoch in range(status_dict["epoch"], self.hps.epochs):
                status_dict["epoch"] = epoch
                initial_step = 0
                if self.shuffle_train:
                    set_seed(epoch)
                    train_dataloader.set_epoch(epoch)
                if first_epoch and "skip_batches" in status_dict:
                    _train_dataloader = accelerator.skip_first_batches(train_dataloader, status_dict["skip_batches"])
                    initial_step = status_dict["skip_batches"]
                else:
                    _train_dataloader = train_dataloader
                torch.cuda.empty_cache()
                model.train()
                for step, batch in tqdm(
                    iterable=enumerate(_train_dataloader, initial_step),
                    total=len(train_dataloader),
                    initial=initial_step,
                    desc=f"Epoch {epoch}/{self.hps.epochs}",
                    unit="batch"
                ):
                    status_dict["epoch_step"] = step
                    CHECKPOINT_EVERY_N_STEPS = _CHECKPOINT_EVERY_N_STEPS and (status_dict["global_step"]+1) % self.checkpoint_every == 0
                    EVALUATION_EVERY_N_STEPS = _EVALUATION_EVERY_N_STEPS and (status_dict["global_step"]+1) % self.evaluate_every_n_steps == 0
                    if (status_dict["global_step"]+1) % self.log_every == 0:
                        status_dict["learning_rate"] = scheduler.get_last_lr()[-1] if scheduler is not None else optimizer.param_groups[0]["lr"]
                        self.monitor.log_learning_rate()
                        self.monitor.log_cpu_utilization()
                        self.monitor.log_gpu_utilization()

                    if accelerator.gradient_accumulation_steps > 1:
                        with accelerator.accumulate(model):
                            self._train_logic(module, optimizer, batch, scheduler, status_dict)
                    else:
                        self._train_logic(module, optimizer, batch, scheduler, status_dict)

                    if CHECKPOINT_EVERY_N_STEPS:
                        self._save_checkpoint(epoch, status_dict["epoch_step"]+1, status_dict, status_dict["epoch_step"]+1)

                    if EVALUATION_EVERY_N_STEPS:
                        self._eval(module, model, val_dataloader, test_dataloader, status_dict, epoch, self.hps.epochs)
                        CHECKPOINT_AFTER_EVALUATION = _CHECKPOINT_AFTER_EVALUATION and (status_dict["evaluations_done"]+1) % self.checkpoint_every == 0
                        if CHECKPOINT_AFTER_EVALUATION:
                            self._save_checkpoint(epoch, status_dict["epoch_step"]+1, status_dict, status_dict["epoch_step"]+1)
                    accelerator.wait_for_everyone()
                
                CHECKPOINT_WHEN_EPOCH_ENDS = ((_CHECKPOINT_WHEN_EPOCH_ENDS and (epoch+1) % self.checkpoint_every == 0) or
                                              (_CHECKPOINT_AFTER_EVALUATION and (status_dict["evaluations_done"]+1) % self.checkpoint_every == 0))

                if self.evaluate_every_n_steps is None:
                    self._eval(module, model, val_dataloader, test_dataloader, status_dict, epoch, self.hps.epochs)

                if CHECKPOINT_WHEN_EPOCH_ENDS:
                    self._save_checkpoint(epoch+1, 0, status_dict, 0)

                first_epoch = False
                gc.collect()

            if self.eval_when_finish and self.evaluate_every_n_steps is not None:
                self._eval(module, model, val_dataloader, test_dataloader, status_dict, epoch, self.hps.epochs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and any(handler in self.handlers for handler in [Handler.CUDA_OUT_OF_MEMORY, Handler.ALL]):
                accelerator.print(time_prefix(), "Forcing checkpointing due to CudaOutOfMemory error.")
                self._save_checkpoint(epoch, status_dict["epoch_step"], status_dict, status_dict["epoch_step"])
            elif any(handler in self.handlers for handler in [Handler.ANY, Handler.ALL]):
                accelerator.print(time_prefix(), "Forcing checkpointing due to a RunTime error.")
                self._save_checkpoint(epoch, status_dict["epoch_step"], status_dict, status_dict["epoch_step"])
            else:
                accelerator.print(e)
                traceback.print_exc()
        except KeyboardInterrupt:
            if any(handler in self.handlers for handler in [Handler.KEYBOARD, Handler.ALL]):
                accelerator.print(time_prefix(), "Forcing checkpointing due to manual keyboard interrupt.")
                self._save_checkpoint(epoch, status_dict["epoch_step"], status_dict, status_dict["epoch_step"])
            else:
                accelerator.print(time_prefix(), "Manual keyboard interrupt.")
                traceback.print_exc()
        except Exception as e:
            if any(handler in self.handlers for handler in [Handler.ANY, Handler.ALL]):
                accelerator.print(time_prefix(), "Forcing checkpointing due to an exception.")
                self._save_checkpoint(epoch, status_dict["epoch_step"], status_dict, status_dict["epoch_step"])
            else:
                accelerator.print(e)
                traceback.print_exc()

        accelerator.end_training()
        if accelerator.num_processes > 1:
            destroy_process_group()
    
    @torch.inference_mode()
    def _eval(self, module, model, val_dataloader, test_dataloader, status_dict, epoch, epochs):
        _test_dataloader = test_dataloader if test_dataloader is not None else val_dataloader
        if val_dataloader is not None and (self.additional_metrics is None or test_dataloader is not None):
            torch.cuda.empty_cache()
            model.eval()
            if self.shuffle_validation:
                set_seed(epoch)
                val_dataloader.set_epoch(epoch)
            for batch in tqdm(
                iterable=val_dataloader,
                total=len(val_dataloader),
                desc=f"Evaluating Epoch {epoch}/{epochs}",
                unit="batch"
            ):
                self._validation_logic(module, batch, status_dict)

            status_dict["evaluations_done"] += 1
            if self.report_loss_after_eval:
                self._log_val_loss(status_dict, total=len(val_dataloader))

        if _test_dataloader is not None and self.additional_metrics is not None:
            # only rank 0 will be in charge of calculating metrics to avoid system overhead
            additional_metrics = ({metric:load(metric, num_process=self.metrics_num_process)
                                   for metric in self.additional_metrics}
                                   if self.additional_metrics is not None else None)
            if additional_metrics is not None and len(additional_metrics) > 0:
                torch.cuda.empty_cache()
                model.eval()
                if self.shuffle_test:
                    set_seed(epoch)
                    _test_dataloader.set_epoch(epoch)
                for batch in tqdm(
                    iterable=_test_dataloader,
                    total=len(_test_dataloader),
                    desc=f"Calculating metrics in Epoch {epoch}/{epochs}",
                    unit="batch"
                ):
                    metrics_dict = self._test_logic(module, batch, additional_metrics, status_dict)

                if accelerator.is_main_process:
                    for metric in additional_metrics.keys():
                        if "loss" in additional_metrics: continue
                        status_dict["additional_metrics"][metric] = additional_metrics[metric].compute()[metric]

                if "loss" in metrics_dict:
                    status_dict["evaluations_done"] += 1
                    if self.report_loss_after_eval:
                        self._log_val_loss(status_dict, total=len(val_dataloader))
                
                self.monitor.log_additional_metrics()

        model.train()
        torch.cuda.empty_cache()

        if self.model_saving is not None:
            self._save_model_on_criteria(model, status_dict)
    
    def _train_logic(self, module, optimizer, batch, scheduler, status_dict):
        loss = module.training_step(batch)
        if loss is None:
            loss = module.step(batch)

        with torch.inference_mode():
            detached_loss = loss.detach()
            self.train_total_loss += detached_loss
            self.train_track_loss += detached_loss

        if (status_dict["global_step"]+1) % self.log_every == 0:
            loss_report = self.train_track_loss / self.log_every
            loss_report = accelerator.reduce(loss_report, reduction="mean").item()
            status_dict["train_loss"] = loss_report
            self.monitor.log_train_loss()
            self.train_track_loss = torch.tensor(0.0, device=accelerator.device) # reset track loss
        
        accelerator.backward(loss)

        if accelerator.sync_gradients and accelerator.is_main_process:
            norm = None
            if self.clip_grad is not None:
                norm = accelerator.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad)
            elif self.monitor.grad_norm:
                norm = self._get_grad_norm()

            if norm is not None and self.monitor.grad_norm:
                status_dict["grad_norm"] = norm
                self.monitor.log_grad_norm()

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=self.set_to_none)

        status_dict["global_step"] += 1
    
    def _validation_logic(self, module, batch, status_dict):
        loss = module.validation_step(batch)
        if loss is None:
            loss = module.step(batch)

        self._track_val_loss(loss, status_dict)

        status_dict["eval_global_step"] += 1

    def _track_val_loss(self, loss, status_dict):
        with torch.inference_mode():
            detached_loss = loss.detach()
            self.val_total_loss += detached_loss
            self.val_track_loss += detached_loss

        # also log it if user wants it immediately
        if not self.report_loss_after_eval:
            if (status_dict["eval_global_step"]+1) % self.log_every == 0:
                self._log_val_loss(status_dict, total=self.log_every)

    def _log_val_loss(self, status_dict, total):
        loss_report = self.val_track_loss / total
        loss_report = accelerator.reduce(loss_report, reduction="mean").item()
        status_dict["validation_loss"] = loss_report
        self.monitor.log_validation_loss()
        self.val_track_loss = torch.tensor(0.0, device=accelerator.device) # reset val loss

    def _test_logic(self, module, batch, additional_metrics, status_dict):
        metrics_dict = module.test_step(batch)
        if metrics_dict is None:
            metrics_dict = module.validation_step(batch)
        
        if "loss" in metrics_dict:
            self._track_val_loss(metrics_dict["loss"], status_dict)
            status_dict["eval_global_step"] += 1

        for metric in additional_metrics.keys():
            if metric == "loss": continue
            # transfer to CPU to avoid GPU memory issues
            predictions = metrics_dict[metric][0]
            predictions = predictions.to("cpu") if predictions.device != "cpu" else predictions
            targets = metrics_dict[metric][1]
            targets = targets.to("cpu") if targets.device != "cpu" else targets

            predictions = accelerator.gather(predictions)
            targets = accelerator.gather(targets)

            if accelerator.is_main_process:
                additional_metrics[metric].add_batch(predictions=predictions, references=targets)

        status_dict["test_global_step"] += 1

        return metrics_dict

    def _save_model(self, model, status_dict, wait_for_everyone=True):
        accelerator.print(time_prefix(), "Saving model...")
        
        with suppress_print_and_warnings(self.verbose):
            PATH_DOES_NOT_EXIST = not os.path.exists(self.model_path) and accelerator.is_main_process
            if PATH_DOES_NOT_EXIST and not wait_for_everyone:
                os.makedirs(self.model_path, exist_ok=True)
            if wait_for_everyone:
                if PATH_DOES_NOT_EXIST:
                    os.makedirs(self.model_path, exist_ok=True)
                accelerator.wait_for_everyone()

            unwrapped_model = accelerator.unwrap_model(model)
            state_dict = unwrapped_model.state_dict() if not self.compile else unwrapped_model._orig_mod.state_dict()
            if hasattr(unwrapped_model, "save_pretrained"):
                unwrapped_model.save_pretrained(
                    self.model_path,
                    is_main_process=accelerator.is_main_process,
                    state_dict=state_dict,
                    max_shard_size=self.max_shard_size,
                    save_function=accelerator.save,
                    safe_serialization=self.safe_serialization
                )
            else:
                accelerator.save(
                    state_dict,
                    f"{self.model_path}/pytorch_model.pt",
                    safe_serialization=self.safe_serialization
                )

            if accelerator.is_main_process:
                save_status(status_dict, to=f"{self.model_path}/{STATUS_PATH}")

        accelerator.print(time_prefix(), "Model saved.")
    
    def _save_model_on_criteria(self, model, status_dict):
        if self.model_saving is None:
            return
        
        accelerator.wait_for_everyone()

        avg_valid_loss = self.val_total_loss / (len(self.val_dataloader) if self.val_dataloader is not None else -1)
        avg_valid_loss = accelerator.reduce(avg_valid_loss, reduction="mean").item()
        avg_train_loss = self.train_total_loss / len(self.train_dataloader)
        avg_train_loss = accelerator.reduce(avg_train_loss, reduction="mean").item()

        if accelerator.is_main_process:
            saving_criteria = {}
            for metric, score in status_dict["additional_metrics"].items():
                best_metric_str = f"best_{metric}"
                if best_metric_str not in status_dict:
                    status_dict[best_metric_str] = -1
                
                prev = status_dict[best_metric_str]
                new = score
                compare = getattr(MetricComparator, metric)
                is_better = compare(new, prev)
                best = new if is_better else prev

                saving_criteria[best_metric_str] = (is_better and
                                                    new < self.model_saving_below and
                                                    new > self.model_saving_above)
                status_dict[best_metric_str] = best

            saving_criteria["best_valid_loss"] = (avg_valid_loss < status_dict["best_valid_loss"] and
                                                avg_valid_loss < self.model_saving_below and
                                                avg_valid_loss > self.model_saving_above)
            saving_criteria["best_train_loss"] = (avg_train_loss < status_dict["best_train_loss"] and
                                                avg_train_loss < self.model_saving_below and
                                                avg_train_loss > self.model_saving_above)
            saving_criteria["always"] = True

            status_dict["best_valid_loss"] = avg_valid_loss if avg_valid_loss < status_dict["best_valid_loss"] else status_dict["best_valid_loss"]
            status_dict["best_train_loss"] = avg_train_loss if avg_train_loss < status_dict["best_train_loss"] else status_dict["best_train_loss"]

            if saving_criteria[self.model_saving]:
                self._save_model(model, status_dict, wait_for_everyone=False)

        self.val_total_loss = torch.tensor(0.0, device=accelerator.device) # reset val total loss
        self.train_total_loss = torch.tensor(0.0, device=accelerator.device) # reset train total loss

    def _save_checkpoint(self, epoch, epoch_step, status_dict, skip_batches):
        accelerator.print(time_prefix(), "Saving checkpoint...")
        with suppress_print_and_warnings(self.verbose):
            if accelerator.is_main_process:
                if not os.path.exists(self.checkpoint):
                    os.makedirs(self.checkpoint, exist_ok=True)
                if not os.path.exists(f"{self.checkpoint}/{CHECKPOINT_PATH}"):
                    os.makedirs(f"{self.checkpoint}/{CHECKPOINT_PATH}", exist_ok=True)
            accelerator.wait_for_everyone()
            accelerator.save_state(f"{self.checkpoint}/{CHECKPOINT_PATH}", safe_serialization=self.safe_serialization)
            if accelerator.is_main_process:
                status = status_dict.copy()
                status["epoch"] = epoch
                status["epoch_step"] = epoch_step
                if (self.checkpoint_strat == "step" or
                    (
                        self.checkpoint_strat == "eval" and
                        self.evaluate_every_n_steps is not None
                    )
                ):
                    status["skip_batches"] = skip_batches
                save_status(status, to=f"{self.checkpoint}/{STATUS_PATH}")
        accelerator.print(time_prefix(), "Checkpoint saved.")

    def _get_optimizer(self, model):
        optimizer = self.hps.optim
        fused_available = "fused" in inspect.signature(optimizer).parameters
        optim_kwargs = self.hps.optim_kwargs
        optim_kwargs["fused"] = fused_available and "cuda" in accelerator.device.type

        filtered_kwargs = self._filter_kwargs(optim_kwargs, optimizer)

        return optimizer(model.parameters(), **filtered_kwargs)
    
    def _scale_learning_rate(self, optimizer):
        for param_group in optimizer.param_groups:
            param_group["lr"] *= accelerator.num_processes

    def _filter_kwargs(self, _kwargs: dict, fn):
        try:
            return {k:v for k,v in _kwargs.items() if k in fn.__init__.__code__.co_varnames}
        except AttributeError:
            signature = inspect.signature(fn)
            parameters = list(signature.parameters.keys())
            return {k:v for k,v in _kwargs.items() if k in parameters}

    def _get_scheduler(self, optimizer, last_epoch, steps_per_epoch, epochs):
        schlr_kwargs = self.hps.scheduler_kwargs
        schlr_kwargs["last_epoch"] = last_epoch
        schlr_kwargs["steps_per_epoch"] = steps_per_epoch // self.grad_accumulation_steps
        total_steps = steps_per_epoch * epochs
        schlr_kwargs["num_training_steps"] = total_steps
        schlr_kwargs["epochs"] = epochs
        if "num_warmup_steps" in schlr_kwargs and isinstance(schlr_kwargs["num_warmup_steps"], float):
            if schlr_kwargs["num_warmup_steps"] < 0.0 or schlr_kwargs["num_warmup_steps"] > 1.0:
                raise ValueError(f"If 'num_warmup_steps' is a ratio (float value), it needs to be a value between 0 and 1.")
            schlr_kwargs["num_warmup_steps"] = round(total_steps * schlr_kwargs["num_warmup_steps"] // self.grad_accumulation_steps)
        elif "warmup_ratio" in schlr_kwargs:
            if schlr_kwargs["warmup_ratio"] > 1.0:
                raise ValueError(f"'warmup_ratio' value in scheduler configuration needs to be a value between 0 and 1.")
            schlr_kwargs["num_warmup_steps"] = round(total_steps * schlr_kwargs["warmup_ratio"] // self.grad_accumulation_steps)

        scheduler = self.hps.scheduler
        filtered_kwargs = self._filter_kwargs(schlr_kwargs, scheduler)

        return scheduler(optimizer, **filtered_kwargs)

    def _initialize_trackers(self):
        if accelerator.is_main_process:
            for logger in self.log_with:
                if logger.tracker == LoggerType.MLFLOW and is_url(self.logging_dir):
                    import mlflow
                    mlflow.set_tracking_uri(self.logging_dir)
                    break

    def _get_collate_fn_pipeline(self):
        def collate_fns(batch):
            for collate_fn in self.collate_fn:
                batch = collate_fn(batch)

            return batch

        return collate_fns
    
    def _get_grad_norm(self) -> float:
        return math.sqrt(sum([torch.norm(p.grad.detach())**2 for p in self.model.parameters() if p.grad is not None]).item())

class Evaluator:
    def __init__(self,
                 collate_fn: Any,
                 batch_size: Optional[int] = 1,
                 dataloader_pin_memory: Optional[int] = True,
                 dataloader_num_workers: Optional[int] = 0
    ):
        """
        Class to evaluate models on validation datasets.

        Args:
            collate_fn (`Any`):
                Collate function to implement when iterating over dataset.
            batch_size (`int`, *optional*, defaults to `1`):
                Batch size to use for evaluation.
        """
        self.collate_fn = collate_fn
        self.batch_size = batch_size
        self.dataloader_pin_memory = dataloader_pin_memory
        self.dataloader_num_workers = dataloader_num_workers

    @torch.inference_mode()
    def eval(self, module: AcceleratorModule, val_dataset: Dataset) -> float:
        """
        Evaluate model on validation dataset and obtain the validation loss value.

        Args:
            module (`AcceleratorModule`):
                `AcceleratorModule` wrapping the actual model to evaluate.
            val_dataset (`Dataset`):
                Validation dataset.
        """
        from torch.utils.data import DataLoader

        model = getattr(module, "model", None)
        if model is None:
            raise AttributeError("'self.model' needs to be declared in the AcceleratorModule class.")
        
        if module._implemented_collate_fn:
            self.collate_fn = module.collate_fn

        dl_kwargs = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "collate_fn": self.collate_fn,
            "pin_memory": self.dataloader_pin_memory
        }
        val_dataloader = DataLoader(val_dataset, **dl_kwargs)

        model, val_dataloader = accelerator.prepare(model, val_dataloader)

        model.eval()
        eval_losses = []
        for batch in tqdm(iterable=val_dataloader, total=len(val_dataloader), desc=f"Evaluating", unit="batch"):
            loss = module.step(batch)
            if loss is None:
                loss = module.validation_step(batch)
            eval_losses.append(loss.item())
        
        avg_eval_loss = np.mean(eval_losses)

        return avg_eval_loss
