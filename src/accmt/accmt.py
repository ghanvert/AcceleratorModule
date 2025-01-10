import inspect
import math
import torch

def allow_tf32(flag=True):
    """Enable or disable the use of TensorFloat32."""
    torch.set_float32_matmul_precision("high" if flag else "highest")

allow_tf32()

from abc import ABC
from accelerate import Accelerator, DataLoaderConfiguration, DistributedType
from accelerate.utils import ProjectConfiguration, InitProcessGroupKwargs, LoggerType, tqdm, set_seed
from .handlers import Handler
import os
import traceback
import torch
import torch.nn as nn
from .utils import get_number_and_unit, is_url, time_prefix, combine_dicts, save_status, read_status, cleanup, operator_map
from .dataloader_samplers import BaseSampler
from .monitor import Monitor
from torch.utils.data import Dataset, DataLoader
from typing_extensions import Any, Optional, Union, override
from .hyperparameters import HyperParameters
from .metrics import Metric
from datetime import timedelta
from .dist_utils import gather_into_single_process

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
        `collate_fn_train` (*optional*):
            Defines the collator function for train DataLoader.
        `collate_fn_val` (*optional*):
            Defines the collator function for validation DataLoader.
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
    _implemented_collate_fn_train = False
    _implemented_collate_fn_val = False
    _accelerator = accelerator
    _log_every: int = 1
    _extended = False
    optimizer = None
    scheduler = None
    device = _accelerator.device
    status_dict: dict = None
    batch_size: Union[int, tuple[int, int]] = None

    @override
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Defines the flow of data."""
    
    @override
    def training_step(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Defines the training logic. Must return a loss tensor (scalar)."""
    
    @override
    def validation_step(self, *args: Any, **kwargs: Any) -> dict:
        """
        Defines the validation logic. Must return a dictionary containing 
        each metric with predictions and targets, and also the loss value in the dictionary.

        Example:
            ```
            # format is ==> "metric": (predictions, targets)
            return {
                "loss": validation_loss_tensor, # (scalar tensor)
                # with additional metrics:
                "accuracy": (accuracy_predictions, accuracy_targets),
                "bleu": (bleu_predictions, bleu_targets)
            }
            ```
        """

    @override
    def collate_fn_train(self, batch: list) -> Any:
        """Defines a collate function for PyTorch train DataLoader."""
        
    @override
    def collate_fn_val(self, batch: list) -> Any:
        """Defines a collate function for PyTorch validation DataLoader."""

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

    @accelerator.on_main_process
    def log(self, values: dict, log_kwargs: dict | None = {}):
        train_or_eval = "global_step" if self.model.training else "eval_global_step"
        if (self.status_dict[train_or_eval]+1) % self._log_every == 0:
            accelerator.log(values, step=self.status_dict[train_or_eval], log_kwargs=log_kwargs)
    
    def __init_subclass__(cls, **kwargs):
        if (
            cls.training_step == AcceleratorModule.training_step and
            cls.validation_step == AcceleratorModule.validation_step
        ):
            raise TypeError(
                "Subclasses of 'Trainer' must override 'training_step' and/or "
                "'validation_step' methods."
            )
        
        if cls.collate_fn_train != AcceleratorModule.collate_fn_train:
            cls._implemented_collate_fn_train = True
        
        if cls.collate_fn_val != AcceleratorModule.collate_fn_val:
            cls._implemented_collate_fn_val = True

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

            def training_step(self, batch):
                return self.model(**batch).loss

            def validation_step(self, batch):
                return {"loss": self.model(**batch).loss}
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

            def training_step(self, batch):
                return self.model(**batch).loss

            def validation_step(self, batch):
                return self.model(**batch).loss
            
        return Module()

class ExtendedAcceleratorModule(AcceleratorModule):
    """
    Extended module from `AcceleratorModule` to enhance `training_step` function. This 
    means that the backpropagation part must be done manually.

    Example:
        ```
        class Module(ExtendedAcceleratorModule):
            # other logic remains the same

            def training_step(self, batch):
                loss = ...
                self.backward(loss)
                self.step_optimizer()
                self.step_scheduler()

                return loss  # loss will only be used to log metrics.
        ```

    NOTE: `grad_accumulation_steps` in `fit` function from `Trainer` will not work. If you want to accumulate gradients 
    and then backpropagate, you may want to make use of `self.status_dict["epoch_step"]`.
    """
    _extended = True

    def backward(self, loss: torch.Tensor, **kwargs):
        """
        Performs backward operation.

        Args:
            `loss` (`torch.Tensor`):
                Scalar loss tensor to backward.
            `kwargs` (`Any`):
                Extra arguments to be passed to 'accelerator.backward' function.
        """
        accelerator.backward(loss, **kwargs)

    def step_optimizer(self):
        self.optimizer.step()

    def step_scheduler(self):
        self.scheduler.step()
    
    def step(self):
        """Step optimizer and scheduler (in that order). If there is no scheduler, it will be ignored."""
        self.step_optimizer()
        if self.scheduler is not None:
            self.step_scheduler()

    def zero_grad(self, set_to_none: bool = True):
        """
        Call optimizer's 'zero_grad' operation to reset gradients.

        Args:
            `set_to_none` (`bool`, *optional*, defaults to `True`):
                Set gradients to `None` instead of `0`.
        """
        self.optimizer.zero_grad(set_to_none=set_to_none)

    @override
    def training_step(self, *args, **kwargs):
        pass

    @override
    def validation_step(self, *args, **kwargs):
        pass


class Trainer:
    """
    Class to implement the training configuration.
    """
    @classmethod
    def from_config(cls,
                    config: Union[str, dict],
                    log_with: Optional[Union[Any, list]] = None,
                    sampler: Optional[Union[Any, list]] = None,
                    collate_fn_train: Optional[Any] = None,
                    collate_fn_val: Optional[Any] = None
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

        return Trainer(**config, log_with=log_with, sampler=sampler, collate_fn_train=collate_fn_train, collate_fn_val=collate_fn_val)

    def __init__(self,
                hps_config: Union[str, dict, HyperParameters],
                model_path: str,
                track_name: Optional[str] = None,
                checkpoint: Optional[str] = None,
                resume: Optional[bool] = None,
                model_saving: Optional[Union[str, list[str]]] = "best_valid_loss",
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
                Enables pin memory option in DataLoader.
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
            kwargs (`Any`, *optional*):
                Extra arguments for specific `init` function in Tracker, e.g. `run_name`, `tags`, etc.
        """
        assert isinstance(hps_config, (str, dict, HyperParameters)), f"'hps_config' needs to be either a string, dictionary or HyperParameters class."
        self.hps = HyperParameters.from_config(hps_config) if isinstance(hps_config, (str, dict)) else hps_config
        self.track_name = track_name
        self.checkpoint = checkpoint if checkpoint is not None else "checkpoint"
        self.checkpoint = f"{model_path}/{self.checkpoint}"
        self.resume = resume if resume is not None else os.path.exists(self.checkpoint) and len(os.listdir(self.checkpoint)) > 0
        self.model_path = model_path
        self.metrics = metrics if isinstance(metrics, list) else [metrics]
        self.metrics = [metric for metric in self.metrics if metric is not None]

        _default_model_savings = set({"best_valid_loss", "best_train_loss", "always"})
        _implemented_metrics = set(f"best_{metric.main_metric}" if not metric.main_metric.startswith("best_") else metric for metric in self.metrics)
        _implemented_metrics.update(_default_model_savings)
        self.model_saving = model_saving if isinstance(model_saving, list) else [model_saving]
        self.model_saving = [ms.lower() for ms in self.model_saving]
        self.model_saving = [f"best_{ms}" if not ms.startswith("best_") else ms for ms in self.model_saving]
        assert all(ms in _implemented_metrics for ms in self.model_saving), f"All 'model_saving' methods should be declared in 'metrics' or be one of {_default_model_savings}."

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
        self.sampler = sampler
        self.model_saving_below = model_saving_below if model_saving_below is not None else float("inf")
        self.model_saving_above = model_saving_above if model_saving_above is not None else float("-inf")
        self.collate_fn_train = self._get_collate_fn_pipeline(collate_fn_train) if isinstance(collate_fn_train, list) else collate_fn_train
        self.collate_fn_val = self._get_collate_fn_pipeline(collate_fn_val) if isinstance(collate_fn_val, list) else collate_fn_val
        self.disable_collate_fn_on_evaluation = disable_collate_fn_on_evaluation
        self.max_shard_size = max_shard_size
        self.safe_serialization = safe_serialization
        self.compile = compile
        self.train_loss_metric_name = train_loss_metric_name
        self.val_loss_metric_name = val_loss_metric_name
        self.dataloader_pin_memory = dataloader_pin_memory
        self.dataloader_num_workers = dataloader_num_workers if dataloader_num_workers is not None else accelerator.num_processes
        self.dataloader_drop_last = dataloader_drop_last
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
        self.cleanup_cache_every_n_steps = cleanup_cache_every_n_steps
        self.init_kwargs = kwargs

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

        self.module = None

    def fit(self,
            module: Union[AcceleratorModule, str, Union[tuple[str, str], tuple[str, Any]]],
            train_dataset: Optional[Dataset] = None,
            val_dataset: Optional[Dataset] = None,
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
            kwargs (`Any`):
                Keyword arguments for `from_pretrained` function for model initialization.
        """
        self.module = module
        self.monitor.additional_metrics = self.metrics is not None and len(self.metrics) > 0
        self.monitor._do_tracking = self.log_with is not None
        self.train_total_loss = torch.tensor(0.0, device=accelerator.device, pin_memory=True)
        self.train_track_loss = torch.tensor(0.0, device=accelerator.device, pin_memory=True)
        self.val_total_loss = torch.tensor(0.0, device=accelerator.device, pin_memory=True)
        self.val_track_loss = torch.tensor(0.0, device=accelerator.device, pin_memory=True)

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
            model.to(accelerator.device) # for optimizer to apply fused when available
            if teacher is not None:
                teacher.to(accelerator.device)
        if self.compile:
            model = torch.compile(model)
            if teacher is not None:
                teacher = torch.compile(teacher)

        if accelerator.distributed_type == DistributedType.FSDP:
            # preparing model before dataloaders is only supported by FSDP apparently, and this is the 
            # recommended setting to prepare training.
            model = accelerator.prepare(model)

        if accelerator.is_main_process:
            os.makedirs(self.model_path, exist_ok=True)

        if self.resume:
            status_dict = read_status(f"{self.checkpoint}/{STATUS_PATH}")
            if "evaluations_done" not in status_dict:
                # in case that ACCMT was updated from < 1.1.0 or 1.2.3 version to a higher one, 
                # this fixes it.
                status_dict["evaluations_done"] = 0
                status_dict["additional_metrics"] = {}
        else:
            status_dict = {
                "best_train_loss": float("inf"),
                "best_valid_loss": float("inf"),
                "epoch": 0,
                "epoch_step": 0,
                "global_step": 0,
                "eval_global_step": 0,
                "evaluations_done": 0,
                "additional_metrics": {}
            }
        module.status_dict = status_dict
        self.monitor._set_extra(accelerator, status_dict, self.train_loss_metric_name, self.val_loss_metric_name)

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
            "drop_last": self.dataloader_drop_last
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
            train_dataloader = DataLoader(train_dataset, shuffle=shuffle_train, sampler=samplers, batch_size=train_batch_size, collate_fn=self.collate_fn_train, **dl_args)

            collate_fn_val = self.collate_fn_val if not self.disable_collate_fn_on_evaluation else None

            val_dataloader = module.get_validation_dataloader()
            if val_dataset is not None and val_dataloader is None:
                val_dataloader = DataLoader(val_dataset, shuffle=self.shuffle_validation, batch_size=val_batch_size, collate_fn=collate_fn_val, **dl_args)
            
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

            scheduler = module.get_scheduler(optimizer, round(len(train_dataloader)/accelerator.num_processes), self.hps.epochs)
            if self.hps.scheduler is not None and scheduler is None:
                scheduler = self._get_scheduler(optimizer, -1, round(len(train_dataloader)/accelerator.num_processes), self.hps.epochs)
                # -1 for last_epoch since Accelerate will take care of recovering the progress

            if self.log_with is not None:
                self._initialize_trackers()

            if accelerator.distributed_type == DistributedType.FSDP:
                train_dataloader, val_dataloader, optimizer, scheduler = accelerator.prepare(
                    train_dataloader, val_dataloader, optimizer, scheduler
                )
            else:
                model, train_dataloader, val_dataloader, optimizer, scheduler = accelerator.prepare(
                    model, train_dataloader, val_dataloader, optimizer, scheduler
                )

            if accelerator.distributed_type != DistributedType.DEEPSPEED and teacher is not None:
                teacher = accelerator.prepare_model(teacher)

            if accelerator.distributed_type == DistributedType.FSDP:
                module.model = model # force module.model to be wrapped to not have problems with dimmensions

            # NOTE: we aren't forcing module.model to be wrapped for other distributed types since we haven't seen any
            # issues with training, and it's actually a little bit faster doing inference with the model directly on the Module class.

            self.model = model

            if scheduler is not None:
                accelerator.register_for_checkpointing(scheduler)

            if self.log_with is not None:
                track_name = self.model_path.split("/")[-1] if self.track_name is None else self.track_name
                init_kwargs = combine_dicts(*[tracker.init(**self.init_kwargs) for tracker in self.log_with])

                config = self.hps.get_config()
                effective_num = self.grad_accumulation_steps * accelerator.num_processes
                config["effective_batch_size"] = (tuple(batch_size*effective_num for batch_size in self.hps.batch_size)
                    if isinstance(self.hps.batch_size, (tuple, list)) else self.hps.batch_size * effective_num
                )
                config["grad_accumulation_steps"] = self.grad_accumulation_steps
                config["num_processes"] = accelerator.num_processes
                accelerator.init_trackers(track_name, config=config, init_kwargs=init_kwargs)

            if self.resume:
                if os.path.exists(self.checkpoint):
                    accelerator.load_state(f"{self.checkpoint}/{CHECKPOINT_PATH}")
                else:
                    raise FileNotFoundError(f"{self.checkpoint} was not found.")

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.module.optimizer = optimizer
        self.module.scheduler = scheduler

        if self.eval_when_start and "evaluations_done" in status_dict and status_dict["evaluations_done"] == 0:
            self._eval(module, model, val_dataloader, status_dict, 0, self.hps.epochs, disable_train_loss=True)
        
        cleanup()
        first_epoch = True
        try:
            for epoch in range(status_dict["epoch"], self.hps.epochs):
                status_dict["epoch"] = epoch
                self.monitor.log_epoch()
                initial_step = 0
                set_seed(epoch)
                if self.shuffle_train:
                    train_dataloader.set_epoch(epoch)
                if first_epoch and "skip_batches" in status_dict:
                    _train_dataloader = accelerator.skip_first_batches(train_dataloader, status_dict["skip_batches"])
                    initial_step = status_dict["skip_batches"]
                else:
                    _train_dataloader = train_dataloader
                cleanup()
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

                    self._train_logic(module, optimizer, batch, scheduler, status_dict)

                    if self.cleanup_cache_every_n_steps is not None and status_dict["global_step"] % self.cleanup_cache_every_n_steps == 0:
                        cleanup()

                    if CHECKPOINT_EVERY_N_STEPS:
                        self._save_checkpoint(epoch, status_dict["epoch_step"]+1, status_dict, status_dict["epoch_step"]+1)

                    if EVALUATION_EVERY_N_STEPS:
                        self._eval(module, model, val_dataloader, status_dict, epoch, self.hps.epochs)
                        CHECKPOINT_AFTER_EVALUATION = _CHECKPOINT_AFTER_EVALUATION and (status_dict["evaluations_done"]+1) % self.checkpoint_every == 0
                        if CHECKPOINT_AFTER_EVALUATION:
                            self._save_checkpoint(epoch, status_dict["epoch_step"]+1, status_dict, status_dict["epoch_step"]+1)
                    accelerator.wait_for_everyone()
                
                CHECKPOINT_WHEN_EPOCH_ENDS = ((_CHECKPOINT_WHEN_EPOCH_ENDS and (epoch+1) % self.checkpoint_every == 0) or
                                              (_CHECKPOINT_AFTER_EVALUATION and (status_dict["evaluations_done"]+1) % self.checkpoint_every == 0))

                if self.evaluate_every_n_steps is None:
                    self._eval(module, model, val_dataloader, status_dict, epoch, self.hps.epochs)

                if CHECKPOINT_WHEN_EPOCH_ENDS:
                    self._save_checkpoint(epoch+1, 0, status_dict, 0)

                first_epoch = False
                cleanup()

            if self.eval_when_finish and self.evaluate_every_n_steps is not None:
                self._eval(module, model, val_dataloader, status_dict, epoch, self.hps.epochs, disable_train_loss=True)
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
    
    @torch.inference_mode()
    def _eval(self, module, model, val_dataloader, status_dict, epoch, epochs, disable_train_loss=False, disable_val_loss=False):
        cleanup()
        model.eval()
        if val_dataloader is not None:
            set_seed(epoch)
            if self.shuffle_validation:
                val_dataloader.set_epoch(epoch)
            for batch in tqdm(
                iterable=val_dataloader,
                total=len(val_dataloader),
                desc=f"Evaluating in Epoch {epoch}/{epochs}",
                unit="batch"
            ):
                self._validation_logic(module, batch, status_dict)

            # only rank 0 will be in charge of calculating metrics to avoid system overhead
            if accelerator.is_main_process:
                for metric in self.metrics:
                    metric_dict = metric._compute()
                    
                    for m, v in metric_dict.items():
                        if not isinstance(v, (float, int)): continue
                        status_dict["additional_metrics"][m] = v

            status_dict["evaluations_done"] += 1
            self._log_val_loss(status_dict, total=len(val_dataloader))
            
            self.monitor.log_additional_metrics()

        model.train()
        cleanup()

        if self.model_saving is not None:
            self._save_model_on_criteria(model, status_dict, disable_train_loss, disable_val_loss)
    
    def _train_logic(self, module, optimizer, batch, scheduler, status_dict):
        loss = module.training_step(batch)
        should_step = (status_dict["epoch_step"]+1) % self.grad_accumulation_steps == 0
        loss = loss / self.grad_accumulation_steps

        with torch.inference_mode():
            detached_loss = loss.detach()
            self.train_total_loss += detached_loss
            self.train_track_loss += detached_loss

        log_every = self.log_every * self.grad_accumulation_steps
        if (status_dict["global_step"]+1) % log_every == 0:
            loss_report = self.train_track_loss / log_every
            loss_report = accelerator.reduce(loss_report, reduction="mean") * self.grad_accumulation_steps
            status_dict["train_loss"] = loss_report.item()
            self.monitor.log_train_loss()
            self.train_track_loss = torch.tensor(0.0, device=accelerator.device) # reset track loss
        
        if accelerator.distributed_type == DistributedType.MULTI_GPU:
            self.model.require_backward_grad_sync = should_step # for gradient sync when using DDP

        if not self.module._extended:
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

        if should_step and not self.module._extended:
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=self.set_to_none)

        status_dict["global_step"] += 1

    def _track_val_loss(self, loss):
        detached_loss = loss.detach()
        self.val_total_loss += detached_loss
        self.val_track_loss += detached_loss

    def _log_val_loss(self, status_dict, total):
        loss_report = self.val_track_loss / total
        loss_report = accelerator.reduce(loss_report, reduction="mean")
        status_dict["validation_loss"] = loss_report.item()
        self.monitor.log_validation_loss()
        self.val_track_loss = torch.tensor(0.0, device=accelerator.device, pin_memory=True) # reset val loss

    def _validation_logic(self, module, batch, status_dict):
        metrics_dict = module.validation_step(batch)
        
        self._track_val_loss(metrics_dict["loss"])

        for metric in self.metrics:
            if metric.name not in metrics_dict:
                predictions, targets = None, None
            else:
                predictions, targets = metrics_dict[metric.name]
            
            predictions = gather_into_single_process(predictions)
            targets = gather_into_single_process(targets)

            if accelerator.is_main_process and predictions is not None and targets is not None:
                # transfer to CPU to avoid GPU memory issues
                predictions = predictions.cpu()
                targets = targets.cpu()
                metric.add_batch(predictions=predictions, references=targets)

        status_dict["eval_global_step"] += 1

    def _save_model(self, model, status_dict, wait_for_everyone=True, model_path=None):
        accelerator.print(time_prefix(), "Saving model...")
        
        model_path = self.model_path if model_path is None else model_path
        os.makedirs(model_path, exist_ok=True) # re-create folder in case it was deleted

        PATH_DOES_NOT_EXIST = not os.path.exists(self.model_path) and accelerator.is_main_process
        if PATH_DOES_NOT_EXIST and not wait_for_everyone:
            os.makedirs(model_path, exist_ok=True)
        if wait_for_everyone:
            if PATH_DOES_NOT_EXIST:
                os.makedirs(model_path, exist_ok=True)
            accelerator.wait_for_everyone()

        unwrapped_model = accelerator.unwrap_model(model)
        state_dict = unwrapped_model.state_dict() if not self.compile else unwrapped_model._orig_mod.state_dict()
        if hasattr(unwrapped_model, "save_pretrained"):
            unwrapped_model.save_pretrained(
                model_path,
                is_main_process=accelerator.is_main_process,
                state_dict=state_dict,
                max_shard_size=self.max_shard_size,
                save_function=accelerator.save,
                safe_serialization=self.safe_serialization
            )
        else:
            accelerator.save(
                state_dict,
                f"{model_path}/pytorch_model.pt",
                safe_serialization=self.safe_serialization
            )

        if accelerator.is_main_process:
            save_status(status_dict, to=f"{model_path}/{STATUS_PATH}")

        accelerator.print(time_prefix(), "Model saved.")
    
    def _save_model_on_criteria(self, model, status_dict, disable_train_loss=False, disable_val_loss=False):
        # use 'disable_train_loss' when evaluating at the beginning
        if self.model_saving is None:
            return
        
        accelerator.wait_for_everyone()

        train_length = self.evaluate_every_n_steps if self.evaluate_every_n_steps is not None else len(self.train_dataloader)

        avg_valid_loss = self.val_total_loss / (len(self.val_dataloader) if self.val_dataloader is not None else -1)
        avg_valid_loss = accelerator.reduce(avg_valid_loss, reduction="mean")
        avg_train_loss = self.train_total_loss / train_length
        avg_train_loss = accelerator.reduce(avg_train_loss, reduction="mean")

        if accelerator.is_main_process:
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

                saving_criteria[best_metric_str] = (is_better and
                                                    new < self.model_saving_below and
                                                    new > self.model_saving_above)
                status_dict[best_metric_str] = best

            if not disable_val_loss:
                saving_criteria["best_valid_loss"] = (avg_valid_loss < status_dict["best_valid_loss"] and
                                                    avg_valid_loss < self.model_saving_below and
                                                    avg_valid_loss > self.model_saving_above)
                status_dict["best_valid_loss"] = avg_valid_loss if avg_valid_loss < status_dict["best_valid_loss"] else status_dict["best_valid_loss"]
                
            if not disable_train_loss:
                saving_criteria["best_train_loss"] = (avg_train_loss < status_dict["best_train_loss"] and
                                                    avg_train_loss < self.model_saving_below and
                                                    avg_train_loss > self.model_saving_above)
                status_dict["best_train_loss"] = avg_train_loss if avg_train_loss < status_dict["best_train_loss"] else status_dict["best_train_loss"]
            
            saving_criteria["always"] = True

            for model_saving in self.model_saving:
                if saving_criteria[model_saving]:
                    model_path = f"{self.model_path}/{model_saving}"
                    self._save_model(model, status_dict, wait_for_everyone=False, model_path=model_path)

        self.val_total_loss = torch.tensor(0.0, device=accelerator.device) # reset val total loss
        self.train_total_loss = torch.tensor(0.0, device=accelerator.device) # reset train total loss

    def _save_checkpoint(self, epoch, epoch_step, status_dict, skip_batches):
        accelerator.print(time_prefix(), "Saving checkpoint...")
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

    def _filter_kwargs(self, _kwargs: dict, fn):
        try:
            return {k:v for k,v in _kwargs.items() if k in fn.__init__.__code__.co_varnames}
        except AttributeError:
            signature = inspect.signature(fn)
            parameters = list(signature.parameters.keys())
            return {k:v for k,v in _kwargs.items() if k in parameters}

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
                raise ValueError(f"If 'num_warmup_steps' is a ratio (float value), it needs to be a value between 0 and 1.")
            schlr_kwargs["num_warmup_steps"] = round(total_steps * schlr_kwargs["num_warmup_steps"])
        elif "warmup_ratio" in schlr_kwargs:
            if schlr_kwargs["warmup_ratio"] > 1.0:
                raise ValueError(f"'warmup_ratio' value in scheduler configuration needs to be a value between 0 and 1.")
            schlr_kwargs["num_warmup_steps"] = round(total_steps * schlr_kwargs["warmup_ratio"])

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

    def _get_collate_fn_pipeline(self, new_collate_fn):
        def collate_fns(batch):
            for collate_fn in new_collate_fn:
                batch = collate_fn(batch)

            return batch

        return collate_fns
    
    def _get_grad_norm(self) -> float:
        return math.sqrt(sum([torch.norm(p.grad.detach())**2 for p in self.model.parameters() if p.grad is not None]).item())
