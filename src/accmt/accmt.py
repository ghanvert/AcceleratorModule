import inspect
import numpy as np
import torch

from abc import ABC
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import ProjectConfiguration, InitProcessGroupKwargs, tqdm
from .tracker import MLFlow
from .events import *
from .config import read, save_status, read_status
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset
from typing import Any
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

init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=86400))
dataloader_config = DataLoaderConfiguration(use_seedable_sampler=True)
accelerator = Accelerator(kwargs_handlers=[init_kwargs], dataloader_config=dataloader_config)

class AcceleratorModule(ABC):
    """
    Super class to define training and validation logic without the need
    to write a training loop.

    The constructor of this class must implement `self.model`, specifying the model
    from `torch.nn.Module`.

    Methods:
        `forward` (optional):
            Defines the flow of data of model. If not implemented, `__call__`
            will not be possible (e.g. `self(...)`).
        `training_step` (optional):
            Defines the training logic. Must return a loss `torch.Tensor` (scalar).
        `validation_step` (optional):
            Defines the validation logic. Must return a loss `torch.Tensor` (scalar).
            If not implemented, no validation will be executed.
        `step` (optional):
            Defines the training and validation logics. This is useful when training
            and validation logics are the same.
            NOTE: Cannot define `step` together with `training_step` and/or
            `validation_step`.
    
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

class Trainer:
    """
    Class to implement the training configuration.
    """

    def __init__(self,
                hps_file_config: str = None,
                checkpoint="checkpoint1",
                resume=False,
                model_path: str = None,
                model_saving="best_valid_loss",
                evaluate_every_n_steps: int = None,
                enable_checkpointing=True,
                checkpoint_strat="epoch",
                checkpoint_every=1,
                logging_dir="logs",
                log_with=False,
                log_every=1,
                grad_accumulation_steps: int = None,
                set_to_none=True,
                shuffle_train=True,
                shuffle_validation=False,
                model_saving_below_loss: float = None,
                collate_fn=None,
                max_shard_size="10GB",
                safe_serialization=False,
                optimizations: list = None,
                fused=True,
                compile=False,
                train_loss_metric_name="train_loss",
                val_loss_metric_name="val_loss"
    ):
        """
        Trainer constructor to set configuration.

        Args:
            hps_file_config (`str`):
                YAML hyperparameters file path.
            checkpoint (`str`, *optional*, defaults to `checkpoint1`):
                Folder path where to save the checkpoint.
            resume (`bool`, *optional*, defaults to `False`):
                Whether to resume from checkpoint or not.
            model_path (`str`, *optional*, defaults to `None`):
                Folder path to save model. If not specified, it will name
                the model path based on the `hps_file_config` name (without the .yaml extension).
            model_saving (`str`, *optional*, defaults to `best_valid_loss`):
                Type of model saving. It can be one of the following values:

                - `"best_valid_loss"`: Saves the model whenever the validation loss is the best recorded.
                - `"best_train_loss"`: Saves the model whenever the training loss is the best recorded.
                - `"always"`: Saves the model always at the end of every epoch.
            evaluate_every_n_steps (`int`, *optional*, defaults to `None`):
                Evaluate model in validation dataset (if implemented) every N steps. If this is set 
                to `None` (default option), evaluation will happen at the end of every epoch.
            enable_checkpointing (`bool`, *optional*, defaults to `True`):
                Whether to save checkpoint or not.
            checkpoint_strat (`str`, *optional*, defaults to `epoch`):
                Strategy to save checkpoint. It can be one of the following values:

                - `"epoch"`: Save a checkpoint at the end of every epoch.
                - `"step"`: Save a checkpoint at a specific step.
                - `"eval"`: Save a checkpoint after evaluation.

                If `checkpoint_strat` is set to `epoch` or `step`, then the checkpoint is done 
                based on the `checkpoint_every` parameter.
            checkpoint_every (`int`, *optional*, defaults to `1`):
                Checkpoint every N steps or epochs (determined by `checkpoint_strat`). 
                If `checkpoint_strat` is set to `eval`, this parameter is not considered.
            logging_dir (`str`, *optional*, defaults to `logs`):
                Path where to save logs to show progress.
            log_with (`str`, *optional*, defaults to `accmt.TensorBoard`):
                Logger to log metrics.
            log_every (`int`, *optional*, defaults to `1`):
                Log every N steps.
            grad_accumulation_steps (`int`, *optional*, defaults to `None`):
                Accumulate gradients for N steps. Useful for training large models and simulate
                large batches when memory is not enough. If set to `None` or `1`, no accumulation will be perfomed.
            set_to_none (`bool`, *optional*, defaults to `True`):
                From PyTorch documentation: "instead of setting to zero, set the grads to None. This will
                in general have lower memory footprint, and can modestly improve performance." Some
                optimizers have a different behaviour if the gradient is 0 or None. See PyTorch docs
                for more information: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
            shuffle_train (`bool`, *optional*, defaults to `True`):
                Whether to shuffle train DataLoader or not.
            shuffle_validation (`bool`, *optional*, defaults to `False`):
                Whether to shuffle validation DataLoader or not.
            model_saving_below_loss (`float`, *optional*, defaults to `float("inf")`):
                Start saving model on this loss (based on `model_saving`). Default is always.
            collate_fn (`function` or `list` of collate functions, *optional*, defaults to `None`):
                Collate function to be implemented in dataloaders. If `module` overrides `collate_fn` from
                `AcceleratorModule` class, then that function will be used instead of the one specified on
                this constructor. If a list of collate functions is given, then the every collate function will affect
                the batch in the given order.
            max_shard_size (`str`, *optional*, defaults to `10GB`):
                Max model shard size to be used.
            safe_serializartion (`bool`, *optional*, defaults to `False`):
                Whether to save model using safe tensors or the traditional PyTorch way. If `True`, some tensors
                will be lost.
            optimizations (`list`, *optional*, defaults to `None`):
                Optimizations from `accmt.optimizations` to be applied during training.
            fused (`bool`, *optional*, defaults to `True`):
                Whether to use fused optimizer when available.
            compile (`bool`, *optional*, defaults to `False`):
                Whether to call `torch.compile` on model.
            train_loss_metric_name (`str`, *optional*, defaults to `train_loss`):
                Metric name for train loss in logs.
            val_loss_metric_name (`str`, *optional*, defaults to `val_loss`):
                Metric name for validation loss in logs.
        """
        assert hps_file_config is not None, "Cannot train without HPS file config."
        self.hps_config = hps_file_config
        self.checkpoint = checkpoint
        self.resume = resume
        self.model_path = model_path
        self.model_saving = model_saving.lower()
        assert self.model_saving in {"best_valid_loss", "best_train_loss", "always"}, f"{self.model_saving} is invalid. Available options are: 'best_valid_loss', 'best_train_loss' and 'always'."
        self.evaluate_every_n_steps = evaluate_every_n_steps
        self.checkpoint_strat = checkpoint_strat.lower()
        assert self.checkpoint_strat in {"epoch", "step", "eval"}, f"{self.checkpoint_strat} is invalid. Available options are: 'epoch', 'step' and 'eval'."
        self.enable_checkpointing = enable_checkpointing
        self.checkpoint_every = checkpoint_every
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
        self.fused = fused
        self.compile = compile
        self.train_loss_metric_name = train_loss_metric_name
        self.val_loss_metric_name = val_loss_metric_name

        self.accelerator = accelerator
        #self.accelerator.gradient_accumulation_steps = grad_accumulation_steps
        self.accelerator.project_configuration = ProjectConfiguration(project_dir=".", logging_dir=logging_dir, total_limit=1)
        
        if log_with is not None:
            if not isinstance(log_with, list): log_with = [log_with]
            self.accelerator.log_with = [tracker.tracker for tracker in log_with]
            self.log_with = [tracker for tracker in log_with]

    def fit(self,
            module: AcceleratorModule,
            train_dataset: Dataset = None,
            val_dataset: Dataset = None
    ):
        """
        Function to train a given `AcceleratorModule`.

        Args:
            module (`AcceleratorModule`):
                `AcceleratorModule` class containig the training logic.
            train_dataset (`torch.utils.data.Dataset`, *optional*, defaults to `None`):
                `Dataset` class from PyTorch containing the train dataset logic.
            val_dataset (`torch.utils.data.Dataset`, *optional*, defaults to `None`):
                `Dataset` class from PyTorch containing the validation dataset logic. If this
                dataset is not specified, then the validation logic of `AcceleratorModule`
                (if specified) will be skipped. If `model_saving` parameter in the constructor is set
                to `best_valid_loss`, this will be converted to `best_train_loss` in the background.
        """
        import os
        import torch

        from torch.utils.data import DataLoader

        model = getattr(module, "model", None)
        if model is None:
            raise AttributeError("'self.model' needs to be declared in the AcceleratorModule class.")
        elif model is not None and not isinstance(model, nn.Module):
            raise ValueError("'self.model' needs to be an instance of 'nn.Module'.")
        
        teacher = getattr(module, "teacher", None)
        if teacher is not None and not isinstance(teacher, nn.Module):
            raise ValueError("'self.teacher' needs to be an instance of 'nn.Module'.")
        
        if torch.cuda.is_available():
            model.to("cuda") # for optimizer to apply fused when available
            if teacher is not None:
                teacher.to("cuda")
        if self.compile:
            model = torch.compile(model)
            if teacher is not None:
                teacher = torch.compile(teacher)
        
        cfg = read(self.hps_config)
        hps = cfg["hps"]
        optim = hps["optim"] if "optim" in hps else None
        schlr = hps["scheduler"] if "scheduler" in hps else None

        os.makedirs(self.model_path, exist_ok=True)

        if self.resume:
            status_dict = read_status(f"{self.checkpoint}/status.json")
        else:
            status_dict = {
                "best_train_loss": float("inf"),
                "best_valid_loss": float("inf"),
                "epoch": 0,
                "epoch_step": 0,
                "global_step": 0,
                "eval_global_step": 0
            }

        train_loss_buffer = None
        val_loss_buffer = None
        if self.log_every > 1 and self.accelerator.is_main_process:
            train_loss_buffer = []
            val_loss_buffer = []

        if module._implemented_collate_fn:
            self.collate_fn = module.collate_fn
        
        train_dataloader = module.get_train_dataloader()
        if train_dataset is not None and train_dataloader is None:
            train_dataloader = DataLoader(train_dataset, batch_size=hps["batch_size"], shuffle=self.shuffle_train, collate_fn=self.collate_fn, pin_memory=True)

        val_dataloader = module.get_validation_dataloader()
        if val_dataset is not None and val_dataloader is None:
            val_dataloader = DataLoader(val_dataset, batch_size=hps["batch_size"], shuffle=self.shuffle_validation, collate_fn=self.collate_fn, pin_memory=True)
        
        # conditionals
        EVALUATION_EVERY_N_STEPS = all([val_dataloader is not None, hasattr(module, "validation_step")]) and self.evaluate_every_n_steps is not None
        CHECKPOINT_EVERY_N_STEPS = self.enable_checkpointing and self.checkpoint_strat == "step"
        CHECKPOINT_AFTER_EVALUATION = self.enable_checkpointing and self.checkpoint_strat == "eval"
        CHECKPOINT_WHEN_EPOCH_ENDS = self.enable_checkpointing and self.checkpoint_strat in {"epoch", "eval"}

        if val_dataloader is None and self.model_saving == "best_valid_loss":
            self.model_saving = "best_train_loss"

        optimizer = module.get_optimizer()
        if optimizer is None:
            optimizer = self._get_optimizer(optim, model)

        scheduler = module.get_scheduler(optimizer, len(train_dataloader), hps["epochs"])
        if schlr is not None and scheduler is None:
            scheduler = self._get_scheduler(schlr, optimizer, -1, len(train_dataloader), hps["epochs"])
            # -1 for last_epoch since Accelerate will take care of recovering the progress

        if self.log_with is not None:
            self._initialize_trackers()

        model, train_dataloader, val_dataloader, optimizer, scheduler, teacher = self.accelerator.prepare(
            model, train_dataloader, val_dataloader, optimizer, scheduler, teacher
        )
        self.model = model

        if scheduler is not None:
            self.accelerator.register_for_checkpointing(scheduler)

        if self.log_with is not None:
            self.accelerator.init_trackers(self.model_path.split("/")[-1], config=hps)

        if self.resume:
            if os.path.exists(self.checkpoint):
                self.accelerator.load_state(f"{self.checkpoint}/{CHECKPOINT_PATH}")
            else:
                raise FileNotFoundError(f"{self.checkpoint} was not found.")

        epochs = hps["epochs"]

        self._apply_start_optimizations()
        first_epoch = True
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
                self._train_logic(module, optimizer, batch, train_losses, scheduler, train_loss_buffer, status_dict)

                if CHECKPOINT_EVERY_N_STEPS and status_dict["global_step"] % self.checkpoint_every == 0:
                    self._save_checkpoint(epoch, status_dict["epoch_step"]+1, status_dict, status_dict["epoch_step"]+1)

                if EVALUATION_EVERY_N_STEPS and status_dict["global_step"] % self.evaluate_every_n_steps == 0:
                    model.eval()
                    eval_losses = []
                    with torch.no_grad():
                        for step, batch in tqdm(
                            iterable=enumerate(val_dataloader, 0),
                            total=len(val_dataloader),
                            desc=f"Evaluating",
                            unit="batch"
                        ):
                            self._validation_logic(module, batch, eval_losses, step, val_loss_buffer, status_dict)
                    
                        self._save_model_on_criteria(model, eval_losses, train_losses, status_dict)
                    
                    if CHECKPOINT_AFTER_EVALUATION:
                        self._save_checkpoint(epoch, status_dict["epoch_step"]+1, status_dict, status_dict["epoch_step"]+1)
                    
                    model.train()
            
            if val_dataloader is not None and self.evaluate_every_n_steps is None:
                model.eval()
                eval_losses = []
                with torch.no_grad():
                    val_dataloader.set_epoch(epoch)
                    for step, batch in tqdm(
                        iterable=enumerate(val_dataloader, 0),
                        total=len(val_dataloader),
                        desc=f"Evaluating Epoch {epoch}/{epochs}",
                        unit="batch"
                    ):
                        self._validation_logic(module, batch, eval_losses, step, val_loss_buffer, status_dict)

            if self.model_saving is not None:
                self._save_model_on_criteria(model, eval_losses, train_losses, status_dict)

            if CHECKPOINT_WHEN_EPOCH_ENDS and (epoch % self.checkpoint_every == 0 or self.checkpoint_strat == "eval"):
                self._save_checkpoint(epoch+1, 0, status_dict, None)
            
            if train_loss_buffer is not None and val_loss_buffer is not None and self.accelerator.is_main_process:
                train_loss_buffer.clear()
                val_loss_buffer.clear()

            first_epoch = False

        self.accelerator.end_training()

    def eval(self, module, val_dataset: Dataset, batch_size=1) -> float:
        from torch.utils.data import DataLoader

        model = getattr(module, "model", None)
        if model is None:
            raise AttributeError("'self.model' needs to be declared in the AcceleratorModule class.")
        
        if module._implemented_collate_fn:
            self.collate_fn = module.collate_fn
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)

        model, val_dataloader = self.accelerator.prepare(model, val_dataloader)

        model.eval()
        eval_losses = []
        with torch.no_grad():
            for batch in tqdm(iterable=val_dataloader, total=len(val_dataloader), desc=f"Evaluating", unit="batch"):
                loss = module.validation_step(batch)
                loss = self.accelerator.gather_for_metrics(loss).cpu().numpy()
                eval_losses.append(np.mean(loss))
        
        avg_eval_loss = np.mean(eval_losses)

        return avg_eval_loss
    
    def _train_logic(self, module, optimizer, batch, train_losses, scheduler, train_loss_buffer, status_dict):
        self._apply_on_batch_optimizations(batch)

        loss = module.training_step(batch, status_dict["global_step"])
        if loss is None:
            loss = module.step(batch, status_dict["global_step"])
        self._apply_on_loss_optimizations(loss)

        loss_item = loss.item()
        train_losses.append(loss_item)
        if train_loss_buffer is not None:
            train_loss_buffer.append(loss_item)
        if (self.accelerator.is_main_process and (status_dict["global_step"] * self.grad_accumulation_steps) % self.log_every == 0):
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
        loss = module.validation_step(batch, status_dict["eval_global_step"])
        if loss is None:
            loss = module.step(batch, status_dict["eval_global_step"])

        loss_item = loss.item()
        eval_losses.append(loss_item)
        if val_loss_buffer is not None and self.accelerator.is_main_process:
            val_loss_buffer.append(loss_item)
        if step % self.log_every == 0 and self.accelerator.is_main_process:
            loss_report = loss_item if val_loss_buffer is None else np.mean(val_loss_buffer)
            if self.log_with is not None:
                self.accelerator.log({self.val_loss_metric_name: loss_report}, step=status_dict["eval_global_step"])
            if val_loss_buffer is not None: val_loss_buffer.clear()

        status_dict["eval_global_step"] += 1

    def _save_model(self, model, status_dict, wait_for_everyone=True):
        if wait_for_everyone:
            self.accelerator.wait_for_everyone()

        self.accelerator.print("Saving model...")
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
            save_status({
                "best_valid_loss": status_dict["best_valid_loss"],
                "best_train_loss": status_dict["best_train_loss"],
            }, to=f"{self.model_path}/status.json")

        self.accelerator.print("Model saved.")
    
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
        self.accelerator.print("Saving checkpoint...")
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
            save_status(status, to=f"{self.checkpoint}/status.json")

    def _get_optimizer(self, optim: dict, model):
        t = optim["type"]
        optim_kwargs = optim.copy()
        del optim_kwargs["type"]
        self._fix_kwargs(optim_kwargs)

        optimizer = OPTIMIZERS[t]
        fused_available = "fused" in inspect.signature(optimizer).parameters
        use_fused = fused_available and "cuda" in self.accelerator.device.type
        if use_fused:
            optim_kwargs["fused"] = self.fused

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
        schlr_kwargs["num_training_steps"] = (steps_per_epoch * epochs) // self.grad_accumulation_steps
        schlr_kwargs["epochs"] = epochs
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
            from .utils import is_url

            for logger in self.log_with:
                if isinstance(logger, MLFlow) and is_url(self.logging_dir):
                    import mlflow
                    mlflow.set_tracking_uri(self.logging_dir)
                    break

    def _get_collate_fn_pipeline(self):
        def collate_fns(batch):
            for collate_fn in self.collate_fn:
                batch = collate_fn(batch)

            return batch

        return collate_fns
