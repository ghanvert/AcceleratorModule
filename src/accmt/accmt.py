import inspect
import numpy as np
import torch

def allow_tf32(flag=True):
    """Enable or disable the use of TensorFloat32."""
    torch.set_float32_matmul_precision("high" if flag else "highest")

allow_tf32()

from abc import ABC
from accelerate import Accelerator, DataLoaderConfiguration, DistributedType
from accelerate.utils import ProjectConfiguration, InitProcessGroupKwargs, LoggerType, tqdm
from .events import *
from .config import read, save_status, read_status
from .handlers import Handler
import traceback
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from .utils import units, get_number_and_unit, is_url, get_num_required_params, time_prefix, combine_dicts
import warnings
from torch.utils.data import Dataset
from typing import Any, Optional, Union
from typing_extensions import override
from transformers import (
    get_cosine_schedule_with_warmup,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_inverse_sqrt_schedule,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    Adafactor
)
from datetime import timedelta

OPTIMIZERS = {
    "Adam": torch.optim.Adam,
    "Adadelta": torch.optim.Adadelta,
    "Adagrad": torch.optim.Adagrad,
    "Adamax": torch.optim.Adamax,
    "AdamW": torch.optim.AdamW,
    "Adafactor": Adafactor,
    "ASGD": torch.optim.ASGD,
    "LBFGS": torch.optim.LBFGS,
    "NAdam": torch.optim.NAdam,
    "RAdam": torch.optim.RAdam,
    "RMSprop": torch.optim.RMSprop,
    "Rprop": torch.optim.Rprop,
    "SGD": torch.optim.SGD,
    "SparseAdam": torch.optim.SparseAdam
}

SCHEDULERS = {
    "StepLR": lr_scheduler.StepLR,
    "LinearLR": lr_scheduler.LinearLR,
    "ExponentialLR": lr_scheduler.ExponentialLR,
    "CosineAnnealingLR": lr_scheduler.CosineAnnealingLR,
    "CyclicLR": lr_scheduler.CyclicLR,
    "OneCycleLR": lr_scheduler.OneCycleLR,
    "CosineAnnealingWarmRestarts": lr_scheduler.CosineAnnealingWarmRestarts,
    "CosineWithWarmup": get_cosine_schedule_with_warmup,
    "Constant": get_constant_schedule,
    "ConstantWithWarmup": get_constant_schedule_with_warmup,
    "CosineWithHardRestartsWithWarmup": get_cosine_with_hard_restarts_schedule_with_warmup,
    "InverseSQRT": get_inverse_sqrt_schedule,
    "LinearWithWarmup": get_linear_schedule_with_warmup,
    "PolynomialDecayWithWarmup": get_polynomial_decay_schedule_with_warmup
}

CHECKPOINT_PATH = "checkpoint"
STATUS_PATH = "status.json"

init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=86400))
dataloader_config = DataLoaderConfiguration(use_seedable_sampler=True)
accelerator = Accelerator(kwargs_handlers=[init_kwargs], dataloader_config=dataloader_config)

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
    def validation_step(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Defines the validation logic. Must return a loss tensor (scalar)."""

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
    def from_config(cls, config: Union[str, dict]):
        """
        Load a configuration from a file or a dictionary.

        Args:
            config (`str` or `dict`):
                Path to a file or dictionary containing kwargs for Trainer constructor. The file can 
                be YAML or JSON.
        """
        assert isinstance(config, (str, dict)), "'config' needs to be either a path to a file, or a dictionary."
        if isinstance(config, str):
            import yaml
            with open(config, "r") as f:
                config = yaml.safe_load(f)

        return Trainer(**config)

    def __init__(self,
                hps_file_config: Union[str, dict],
                track_name: Optional[str] = None,
                checkpoint: Optional[str] = None,
                resume: Optional[bool] = False,
                model_path: Optional[str] = None,
                model_saving: Optional[str] = "best_valid_loss",
                evaluate_every_n_steps: Optional[int] = None,
                checkpoint_every: Optional[str] = "epoch",
                logging_dir: Optional[str] = "logs",
                log_with: Optional[Union[Any, list[Any]]] = None,
                log_every: Optional[int] = 1,
                grad_accumulation_steps: Optional[int] = None,
                set_to_none: Optional[bool] = True,
                shuffle_train: Optional[bool] = True,
                shuffle_validation: Optional[bool] = False,
                model_saving_below_loss: Optional[float] = None,
                collate_fn: Optional[Any] = None,
                max_shard_size: Optional[str] = "10GB",
                safe_serialization: Optional[bool] = False,
                optimizations: Optional[list[Any]] = None,
                compile: Optional[bool] = False,
                train_loss_metric_name: Optional[str] = "train_loss",
                val_loss_metric_name: Optional[str] = "val_loss",
                dataloader_pin_memory: Optional[bool] = True,
                dataloader_num_workers: Optional[int] = 0,
                report_loss_after_eval: Optional[bool] = True,
                handlers: Optional[Union[list, Any]] = None,
                eval_when_finish: Optional[bool] = True,
                eval_when_start: Optional[bool] = False,
                verbose: Optional[bool] = True,
                **kwargs: Optional[Any]
    ):
        """
        Trainer constructor to set configuration.

        Args:
            hps_file_config (`str` or `dict`):
                YAML hyperparameters file path or dictionary.
            track_name (`str`, *optional*, defaults to `None`):
                Track name for trackers. If set to `None` (default), the track name will be 
                the model's folder name.
            checkpoint (`str`, *optional*, defaults to `None`):
                Path where to save the checkpoint. Path by default is going to be of type: 
                'checkpoint-MODEL_PATH_NAME'.
            resume (`bool`, *optional*, defaults to `False`):
                Whether to resume from checkpoint.
            model_path (`str`, *optional*, defaults to `None`):
                Path to save model. If not specified, it will name
                the model path based on the `hps_file_config` name (without the .yaml extension).
            model_saving (`str`, *optional*, defaults to `best_valid_loss`):
                Type of model saving. It can be one of the following values:

                - `"best_valid_loss"`: Saves the model whenever the validation loss is the best recorded.
                - `"best_train_loss"`: Saves the model whenever the training loss is the best recorded.
                - `"always"`: Saves the model always at the end of every evaluation.

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
            set_to_none (`bool`, *optional*, defaults to `True`):
                From PyTorch documentation: "instead of setting to zero, set the grads to None. This will
                in general have lower memory footprint, and can modestly improve performance." Some
                optimizers have a different behaviour if the gradient is 0 or None. See PyTorch docs
                for more information: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
            shuffle_train (`bool`, *optional*, defaults to `True`):
                Whether to shuffle train DataLoader.
            shuffle_validation (`bool`, *optional*, defaults to `False`):
                Whether to shuffle validation DataLoader.
            model_saving_below_loss (`float`, *optional*, defaults to `float("inf")`):
                Start saving model on this loss (based on `model_saving`). Default is always saving.
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
            optimizations (`list`, *optional*, defaults to `None`) - beta:
                Optimizations from `accmt.optimizations` to be applied during training.

                NOTE: some of these optimizations are not tested, so you might encounter some errors.
            compile (`bool`, *optional*, defaults to `False`):
                Whether to call `torch.compile` on model (and teacher, if implemented).
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
            verbose (`bool`, *optional*, defaults to `True`):
                Enable prints when checkpointing and saving model.
            kwargs (`Any`, *optional*):
                Extra arguments for specific `init` function in Tracker, e.g. `run_name`, `tags`, etc.
        """
        self.hps_config = hps_file_config
        self.track_name = track_name
        self.checkpoint = checkpoint if checkpoint is not None else f"checkpoint-{model_path.split('/')[-1]}"
        self.resume = resume
        self.model_path = model_path
        self.model_saving = model_saving.lower()
        assert self.model_saving in {"best_valid_loss", "best_train_loss", "always"}, f"{self.model_saving} is invalid. Available options are: 'best_valid_loss', 'best_train_loss' and 'always'."
        self.evaluate_every_n_steps = evaluate_every_n_steps
        assert self.checkpoint_strat in units.keys(), f"{self.checkpoint_strat} is invalid. Available options are: 'epoch', 'step' and 'eval'."
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
        self.set_to_none = set_to_none
        self.shuffle_train = shuffle_train
        self.shuffle_validation = shuffle_validation
        self.model_saving_below_loss = model_saving_below_loss if model_saving_below_loss is not None else float("inf")
        self.collate_fn = self._get_collate_fn_pipeline() if isinstance(collate_fn, list) else collate_fn
        self.max_shard_size = max_shard_size
        self.safe_serialization = safe_serialization
        self.optimizations = optimizations if optimizations is not None else []
        self.compile = compile
        self.train_loss_metric_name = train_loss_metric_name
        self.val_loss_metric_name = val_loss_metric_name
        self.dataloader_pin_memory = dataloader_pin_memory
        self.dataloader_num_workers = dataloader_num_workers
        self.report_loss_after_eval = report_loss_after_eval
        self.handlers = handlers if isinstance(handlers, list) else [handlers]
        if self.handlers[0] is not None: raise NotImplementedError("'handlers' argument is not yet fully implemented.")
        self.eval_when_finish = eval_when_finish
        self.eval_when_start = eval_when_start
        self.verbose = verbose
        self.init_kwargs = kwargs

        self.accelerator = accelerator
        if isinstance(grad_accumulation_steps, int) and grad_accumulation_steps > 1:
            self.accelerator.gradient_accumulation_steps = grad_accumulation_steps
        self.accelerator.project_configuration = ProjectConfiguration(project_dir=".", logging_dir=logging_dir, total_limit=1)

        if log_with is not None:
            if not isinstance(log_with, list): log_with = [log_with]
            self.accelerator.log_with = [tracker.tracker for tracker in log_with]
            self.log_with = [tracker for tracker in log_with]

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
        import os
        import torch

        from torch.utils.data import DataLoader

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
        
        if torch.cuda.is_available():
            model.to(self.accelerator.device) # for optimizer to apply fused when available
            if teacher is not None:
                teacher.to(self.accelerator.device)
        if self.compile:
            model = torch.compile(model)
            if teacher is not None:
                teacher = torch.compile(teacher)

        if self.accelerator.distributed_type == DistributedType.FSDP:
            model = self.accelerator.prepare(model)
        
        cfg = read(self.hps_config) if isinstance(self.hps_config, str) else self.hps_config
        if self.model_path is None:
            self.model_path = cfg["version"]
        hps = cfg["hps"]
        optim = hps["optim"] if "optim" in hps else None
        schlr = hps["scheduler"] if "scheduler" in hps else None

        os.makedirs(self.model_path, exist_ok=True)

        if self.resume:
            status_dict = read_status(f"{self.checkpoint}/{STATUS_PATH}")
            if "evaluations_done" not in status_dict:
                # in case that ACCMT was updated from < 1.1.0 version to a higher one, 
                # this fixes it.
                status_dict["evaluations_done"] = 0
        else:
            status_dict = {
                "best_train_loss": float("inf"),
                "best_valid_loss": float("inf"),
                "epoch": 0,
                "epoch_step": 0,
                "global_step": 0,
                "eval_global_step": 0,
                "evaluations_done": 0
            }

        train_loss_buffer = None
        val_loss_buffer = None
        if self.log_every > 1 and self.accelerator.is_main_process:
            train_loss_buffer = []
            val_loss_buffer = []

        if module._implemented_collate_fn:
            self.collate_fn = module.collate_fn

        dl_args = {
            "batch_size": hps["batch_size"],
            "collate_fn": self.collate_fn,
            "pin_memory": self.dataloader_pin_memory,
            "num_workers": self.dataloader_num_workers
        }

        train_dataloader = module.get_train_dataloader()
        if train_dataset is not None and train_dataloader is None:
            train_dataloader = DataLoader(train_dataset, shuffle=self.shuffle_train, **dl_args)

        val_dataloader = module.get_validation_dataloader()
        if val_dataset is not None and val_dataloader is None:
            val_dataloader = DataLoader(val_dataset, shuffle=self.shuffle_validation, **dl_args)
        
        # conditionals
        _EVALUATION_EVERY_N_STEPS = all([val_dataloader is not None, hasattr(module, "validation_step")]) and self.evaluate_every_n_steps is not None
        _CHECKPOINT_EVERY_N_STEPS = self.enable_checkpointing and self.checkpoint_strat == "step"
        _CHECKPOINT_AFTER_EVALUATION = self.enable_checkpointing and self.checkpoint_strat == "eval"
        _CHECKPOINT_WHEN_EPOCH_ENDS = self.enable_checkpointing and self.checkpoint_strat in {"epoch", "eval"}

        if val_dataloader is None and self.model_saving == "best_valid_loss":
            self.model_saving = "best_train_loss"

        optimizer = module.get_optimizer()
        if optimizer is None:
            optimizer = self._get_optimizer(optim, model)

        scheduler = module.get_scheduler(optimizer, len(train_dataloader), hps["epochs"])
        if schlr is not None and scheduler is None:
            scheduler = self._get_scheduler(schlr, optimizer, -1, len(val_dataloader), hps["epochs"])
            # -1 for last_epoch since Accelerate will take care of recovering the progress

        if self.log_with is not None:
            self._initialize_trackers()

        if self.accelerator.distributed_type == DistributedType.FSDP:
            train_dataloader, val_dataloader, optimizer, scheduler, teacher = self.accelerator.prepare(
                train_dataloader, val_dataloader, optimizer, scheduler, teacher
            )
            module.model = model
        else:
            model, train_dataloader, val_dataloader, optimizer, scheduler, teacher = self.accelerator.prepare(
                model, train_dataloader, val_dataloader, optimizer, scheduler, teacher
            )
        self.model = model

        if scheduler is not None:
            self.accelerator.register_for_checkpointing(scheduler)

        if self.log_with is not None:
            track_name = self.model_path.split("/")[-1] if self.track_name is None else self.track_name
            init_kwargs = combine_dicts(*[tracker.init(**self.init_kwargs) for tracker in self.log_with])
            self.accelerator.init_trackers(track_name, config=hps, init_kwargs=init_kwargs)

        if self.resume:
            if os.path.exists(self.checkpoint):
                self.accelerator.load_state(f"{self.checkpoint}/{CHECKPOINT_PATH}")
            else:
                raise FileNotFoundError(f"{self.checkpoint} was not found.")

        epochs = hps["epochs"]

        self._apply_start_optimizations()

        if self.eval_when_start and "evaluations_done" in status_dict and status_dict["evaluations_done"] == 0:
            self._eval(module, model, val_dataloader, val_loss_buffer, [], status_dict, 0, epochs)

        first_epoch = True
        try:
            for epoch in range(status_dict["epoch"], epochs):
                status_dict["epoch"] = epoch
                initial_step = 0
                train_dataloader.set_epoch(epoch)
                if first_epoch and "skip_batches" in status_dict:
                    _train_dataloader = accelerator.skip_first_batches(train_dataloader, status_dict["skip_batches"])
                    initial_step = status_dict["skip_batches"]
                else:
                    _train_dataloader = train_dataloader
                torch.cuda.empty_cache()
                model.train()
                train_losses = []
                for step, batch in tqdm(
                    iterable=enumerate(_train_dataloader, initial_step),
                    total=len(train_dataloader),
                    initial=initial_step,
                    desc=f"Epoch {epoch}/{epochs}",
                    unit="batch"
                ):
                    status_dict["epoch_step"] = step
                    CHECKPOINT_EVERY_N_STEPS = _CHECKPOINT_EVERY_N_STEPS and (status_dict["global_step"]+1) % self.checkpoint_every == 0
                    EVALUATION_EVERY_N_STEPS = _EVALUATION_EVERY_N_STEPS and (status_dict["global_step"]+1) % self.evaluate_every_n_steps == 0

                    self._train_logic(module, optimizer, batch, train_losses, scheduler, train_loss_buffer, status_dict)

                    if CHECKPOINT_EVERY_N_STEPS:
                        self._save_checkpoint(epoch, status_dict["epoch_step"]+1, status_dict, status_dict["epoch_step"]+1)

                    if EVALUATION_EVERY_N_STEPS:
                        self._eval(module, model, val_dataloader, val_loss_buffer, train_losses, status_dict, epoch, epochs)
                        CHECKPOINT_AFTER_EVALUATION = _CHECKPOINT_AFTER_EVALUATION and (status_dict["evaluations_done"]+1) % self.checkpoint_every == 0
                        if CHECKPOINT_AFTER_EVALUATION:
                            self._save_checkpoint(epoch, status_dict["epoch_step"]+1, status_dict, status_dict["epoch_step"]+1)
                
                CHECKPOINT_WHEN_EPOCH_ENDS = ((_CHECKPOINT_WHEN_EPOCH_ENDS and (epoch+1) % self.checkpoint_every == 0) or
                                              (_CHECKPOINT_AFTER_EVALUATION and (status_dict["evaluations_done"]+1) % self.checkpoint_every == 0))

                if self.evaluate_every_n_steps is None:
                    self._eval(module, model, val_dataloader, val_loss_buffer, train_losses, status_dict, epoch, epochs)

                if CHECKPOINT_WHEN_EPOCH_ENDS:
                    self._save_checkpoint(epoch+1, 0, status_dict, None)
                
                if train_loss_buffer is not None and val_loss_buffer is not None and self.accelerator.is_main_process:
                    train_loss_buffer.clear()
                    val_loss_buffer.clear()

                first_epoch = False

            if self.eval_when_finish and self.evaluate_every_n_steps is not None:
                self._eval(module, model, val_dataloader, val_loss_buffer, train_losses, status_dict, epoch, epochs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and any(handler in self.handlers for handler in [Handler.CUDA_OUT_OF_MEMORY, Handler.ALL]):
                self.accelerator.print(time_prefix(), "Forcing checkpointing due to CudaOutOfMemory error.")
                self._save_checkpoint(epoch, status_dict["epoch_step"], status_dict, status_dict["epoch_step"])
            elif any(handler in self.handlers for handler in [Handler.ANY, Handler.ALL]):
                self.accelerator.print(time_prefix(), "Forcing checkpointing due to a RunTime error.")
                self._save_checkpoint(epoch, status_dict["epoch_step"], status_dict, status_dict["epoch_step"])
            else:
                self.accelerator.print(e)
                traceback.print_exc()
        except KeyboardInterrupt:
            if any(handler in self.handlers for handler in [Handler.KEYBOARD, Handler.ALL]):
                self.accelerator.print(time_prefix(), "Forcing checkpointing due to manual keyboard interrupt.")
                self._save_checkpoint(epoch, status_dict["epoch_step"], status_dict, status_dict["epoch_step"])
            else:
                self.accelerator.print(time_prefix(), "Manual keyboard interrupt.")
                traceback.print_exc()
        except Exception as e:
            if any(handler in self.handlers for handler in [Handler.ANY, Handler.ALL]):
                self.accelerator.print(time_prefix(), "Forcing checkpointing due to an exception.")
                self._save_checkpoint(epoch, status_dict["epoch_step"], status_dict, status_dict["epoch_step"])
            else:
                self.accelerator.print(e)
                traceback.print_exc()

        self.accelerator.end_training()

    @torch.no_grad()
    def eval(self, module: AcceleratorModule, val_dataset: Dataset, batch_size: int = 1) -> float:
        """
        Evaluate model on validation dataset and obtain the validation loss value.

        Args:
            module (`AcceleratorModule`):
                `AcceleratorModule` wrapping the actual model to evaluate.
            val_dataset (`Dataset`):
                Validation dataset.
            batch_size (`int`, *optional*, defaults to `1`):
                Batch size to use for evaluation.
        """
        from torch.utils.data import DataLoader

        model = getattr(module, "model", None)
        if model is None:
            raise AttributeError("'self.model' needs to be declared in the AcceleratorModule class.")
        
        if module._implemented_collate_fn:
            self.collate_fn = module.collate_fn
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn, pin_memory=True)

        model, val_dataloader = self.accelerator.prepare(model, val_dataloader)

        model.eval()
        eval_losses = []
        for batch in tqdm(iterable=val_dataloader, total=len(val_dataloader), desc=f"Evaluating", unit="batch"):
            loss = module.step(batch)
            if loss is None:
                loss = module.validation_step(batch)
            eval_losses.append(loss.item())
        
        avg_eval_loss = np.mean(eval_losses)

        return avg_eval_loss
    
    @torch.no_grad()
    def _eval(self, module, model, val_dataloader, val_loss_buffer, train_losses, status_dict, epoch, epochs):
        torch.cuda.empty_cache()
        eval_losses = []
        if val_dataloader is not None:
            model.eval()
            val_dataloader.set_epoch(epoch)
            for step, batch in tqdm(
                iterable=enumerate(val_dataloader, 0),
                total=len(val_dataloader),
                desc=f"Evaluating Epoch {epoch}/{epochs}",
                unit="batch"
            ):
                self._validation_logic(module, batch, eval_losses, step, val_loss_buffer, status_dict)

            status_dict["evaluations_done"] += 1
            if self.report_loss_after_eval and self.log_with is not None:
                val_loss = np.mean(eval_losses)
                self.accelerator.log({self.val_loss_metric_name: val_loss}, step=status_dict["global_step"])

            model.train()
            torch.cuda.empty_cache()

        if self.model_saving is not None:
            self._save_model_on_criteria(model, eval_losses, train_losses, status_dict)
    
    def _train_logic(self, module, optimizer, batch, train_losses, scheduler, train_loss_buffer, status_dict):
        self._apply_on_batch_optimizations(batch)

        num_params = get_num_required_params(module.training_step)
        loss = module.training_step(batch, status_dict) if num_params == 2 else module.training_step(batch)
        if loss is None:
            num_params = get_num_required_params(module.step)
            loss = module.step(batch, status_dict) if num_params == 2 else module.step(batch)
        self._apply_on_loss_optimizations(loss)

        loss_item = loss.item()
        train_losses.append(loss_item)
        if train_loss_buffer is not None:
            train_loss_buffer.append(loss_item)
        if (self.accelerator.is_main_process and ((status_dict["global_step"]+1) * self.grad_accumulation_steps) % self.log_every == 0):
            loss_report = loss_item if train_loss_buffer is None else np.mean(train_loss_buffer)
            if self.log_with is not None:
                self.accelerator.log({self.train_loss_metric_name: loss_report}, step=status_dict["global_step"])
            if train_loss_buffer is not None: train_loss_buffer.clear()
        
        self._apply_before_backward_optimizations(self.model.parameters())
        self.accelerator.backward(loss)
        self._apply_after_backward_optimizations(self.model.parameters())

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=self.set_to_none)

        status_dict["global_step"] += 1
    
    def _validation_logic(self, module, batch, eval_losses, step, val_loss_buffer, status_dict):
        num_params = get_num_required_params(module.validation_step)
        loss = module.validation_step(batch, status_dict) if num_params == 2 else module.validation_step(batch)
        if loss is None:
            num_params = get_num_required_params(module.step)
            loss = module.step(batch, status_dict) if num_params == 2 else module.step(batch)

        loss_item = loss.item()
        eval_losses.append(loss_item)
        if val_loss_buffer is not None and self.accelerator.is_main_process and not self.report_loss_after_eval:
            val_loss_buffer.append(loss_item)
        if (step+1) % self.log_every == 0 and self.accelerator.is_main_process:
            if not self.report_loss_after_eval:
                loss_report = loss_item if val_loss_buffer is None else np.mean(val_loss_buffer)
                if self.log_with is not None:
                    self.accelerator.log({self.val_loss_metric_name: loss_report}, step=status_dict["eval_global_step"])
            if val_loss_buffer is not None: val_loss_buffer.clear()

        status_dict["eval_global_step"] += 1

    def _save_model(self, model, status_dict, wait_for_everyone=True):
        if wait_for_everyone:
            self.accelerator.wait_for_everyone()

        if self.verbose: self.accelerator.print(time_prefix(), "Saving model...")
        unwrapped_model = self.accelerator.unwrap_model(model)
        state_dict = unwrapped_model.state_dict() if not self.compile else unwrapped_model._orig_mod.state_dict()
        if hasattr(unwrapped_model, "save_pretrained"):
            unwrapped_model.save_pretrained(
                self.model_path,
                is_main_process=self.accelerator.is_main_process,
                state_dict=state_dict,
                max_shard_size=self.max_shard_size,
                save_function=self.accelerator.save,
                safe_serialization=self.safe_serialization
            )
        else:
            self.accelerator.save(
                state_dict,
                f"{self.model_path}/pytorch_model.pt",
                safe_serialization=self.safe_serialization
            )

        if self.accelerator.is_main_process:
            save_status(status_dict, to=f"{self.model_path}/{STATUS_PATH}")

        if self.verbose: self.accelerator.print(time_prefix(), "Model saved.")
    
    def _save_model_on_criteria(self, model, eval_losses, train_losses, status_dict):
        if self.model_saving is None:
            return
        
        self.accelerator.wait_for_everyone()

        avg_valid_loss = np.mean(eval_losses)
        avg_train_loss = np.mean(train_losses)

        saving_criteria = {
            "best_valid_loss": avg_valid_loss < status_dict["best_valid_loss"] and avg_valid_loss < self.model_saving_below_loss,
            "best_train_loss": avg_train_loss < status_dict["best_train_loss"] and avg_train_loss < self.model_saving_below_loss,
            "always": True
        }

        status_dict["best_valid_loss"] = avg_valid_loss if avg_valid_loss < status_dict["best_valid_loss"] else status_dict["best_valid_loss"]
        status_dict["best_train_loss"] = avg_train_loss if avg_train_loss < status_dict["best_train_loss"] else status_dict["best_train_loss"]

        if saving_criteria[self.model_saving]:
            self._save_model(model, status_dict, wait_for_everyone=False)
    
    def _fix_kwargs(self, dictionary: dict):
        for k, v in dictionary.items():
            if isinstance(v, str):
                try:
                    dictionary[k] = float(v)
                except ValueError:
                    continue

    def _save_checkpoint(self, epoch, epoch_step, status_dict, skip_batches):
        self.accelerator.wait_for_everyone()
        if self.verbose: self.accelerator.print(time_prefix(), "Saving checkpoint...")
        self.accelerator.save_state(f"{self.checkpoint}/{CHECKPOINT_PATH}", safe_serialization=self.safe_serialization)
        if self.accelerator.is_main_process:
            status = status_dict.copy()
            status["epoch"] = epoch
            status["epoch_step"] = epoch_step
            if (self.checkpoint_strat == "step" or
                (self.checkpoint_strat == "eval" and self.evaluate_every_n_steps is not None) and
                skip_batches is not None
            ):
                status["skip_batches"] = skip_batches
            save_status(status, to=f"{self.checkpoint}/{STATUS_PATH}")

    def _get_optimizer(self, optim: dict, model):
        t = optim["type"]
        optim_kwargs = optim.copy()
        del optim_kwargs["type"]
        self._fix_kwargs(optim_kwargs)

        optimizer = OPTIMIZERS[t]
        fused_available = "fused" in inspect.signature(optimizer).parameters
        optim_kwargs["fused"] = fused_available and "cuda" in self.accelerator.device.type

        return optimizer(model.parameters(), **optim_kwargs)

    def _filter_kwargs(self, _kwargs: dict, fn):
        try:
            return {k:v for k,v in _kwargs.items() if k in fn.__init__.__code__.co_varnames}
        except AttributeError:
            signature = inspect.signature(fn)
            parameters = list(signature.parameters.keys())
            return {k:v for k,v in _kwargs.items() if k in parameters}

    def _get_scheduler(self, schlr: dict, optimizer, last_epoch, steps_per_epoch, epochs):
        t = schlr["type"]
        schlr_kwargs = schlr.copy()
        del schlr_kwargs["type"]
        self._fix_kwargs(schlr_kwargs)

        schlr_kwargs["last_epoch"] = last_epoch
        schlr_kwargs["steps_per_epoch"] = steps_per_epoch
        total_steps = steps_per_epoch * epochs
        schlr_kwargs["num_training_steps"] = total_steps // self.grad_accumulation_steps
        schlr_kwargs["epochs"] = epochs
        if "num_warmup_steps" in schlr_kwargs and isinstance(schlr_kwargs["num_warmup_steps"], float):
            if schlr_kwargs["num_warmup_steps"] < 0.0 or schlr_kwargs["num_warmup_steps"] > 1.0:
                raise ValueError(f"If 'num_warmup_steps' is a ratio (float value), it needs to be a value between 0 and 1.")
            schlr_kwargs["num_warmup_steps"] = round(total_steps * schlr_kwargs["num_warmup_steps"] // self.grad_accumulation_steps)
        elif "warmup_ratio" in schlr_kwargs:
            if schlr_kwargs["warmup_ratio"] > 1.0:
                raise ValueError(f"'warmup_ratio' value in scheduler configuration needs to be a value between 0 and 1.")
            schlr_kwargs["num_warmup_steps"] = round(total_steps * schlr_kwargs["warmup_ratio"] // self.grad_accumulation_steps)
        filtered_kwargs = self._filter_kwargs(schlr_kwargs, SCHEDULERS[t])

        return SCHEDULERS[t](optimizer, **filtered_kwargs)

    def _apply_start_optimizations(self):
        if self.accelerator.is_main_process:
            for optimization in self.optimizations:
                type = optimization.__class__.__bases__[0]
                if isinstance(type, Start):
                    optimization()

    def _apply_epoch_start_optimizations(self):
        if self.accelerator.is_main_process:
            for optimization in self.optimizations:
                type = optimization.__class__.__bases__[0]
                if isinstance(type, EpochStart):
                    optimization()

    def _apply_epoch_end_optimizations(self):
        if self.accelerator.is_main_process:
            for optimization in self.optimizations:
                type = optimization.__class__.__bases__[0]
                if isinstance(type, EpochEnd):
                    optimization()

    def _apply_on_batch_optimizations(self, batch):
        for optimization in self.optimizations:
            type = optimization.__class__.__bases__[0]
            if isinstance(type, OnBatch):
                optimization(batch)

    def _apply_on_loss_optimizations(self, loss):
        for optimization in self.optimizations:
            type = optimization.__class__.__bases__[0]
            if isinstance(type, OnLoss):
                optimization(loss)

    def _apply_before_backward_optimizations(self, parameters):
        for optimization in self.optimizations:
            type = optimization.__class__.__bases__[0]
            if isinstance(type, BeforeBackward):
                optimization(parameters)

    def _apply_after_backward_optimizations(self, parameters):
        for optimization in self.optimizations:
            type = optimization.__class__.__bases__[0]
            if isinstance(type, AfterBackward):
                optimization(parameters)

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
