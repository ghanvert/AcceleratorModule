import numpy as np
import torch

from abc import ABC
from accelerate import Accelerator
from accelerate.utils import LoggerType, ProjectConfiguration, tqdm
from .config import read, save_status, read_status
import torch.optim
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
                checkpoint = "checkpoint1",
                resume = False,
                model_path: str = None,
                model_saving = "best_valid_loss",
                enable_checkpointing = True,
                checkpoint_every = 1,
                logging_dir = "logs",
                log_with = LoggerType.TENSORBOARD,
                log_every = 1,
                grad_accumulation_steps=None,
                set_to_none=True,
                shuffle_train=True,
                shuffle_validation=False,
                model_saving_below_loss=float("inf"),
                collate_fn=None,
                max_shard_size="10GB",
                safe_serialization=False
    ):
        """
        Trainer constructor to set configuration.

        Args:
            hps_file_config (`str`):
                YAML hyperparameters file path.
            checkpoint (`str`, *optional*, default to `checkpoint1`):
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
            
            enable_checkpointing (`bool`, *optional*, defaults to `True`):
                Whether to save checkpoint or not.
            checkpoint_every (`int`, *optional*, defaults to `1`):
                Checkpoint every N steps. Only works if `enable_checkpointing` is set to `True`.
            logging_dir (`str`, *optional*, defaults to `logs`):
                Path where to save logs to show progress.
            log_with (`str`, *optional*, defaults to `LoggerType.TENSORBOARD`):
                `LoggerType` to log progress.
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
            collate_fn (`function`, *optional*, defaults to `None`):
                Collate function to be implemented in dataloaders. If `module` overrides `collate_fn` from
                `AcceleratorModule` class, then that function will be used instead of the one specified on
                this constructor.
            max_shard_size (`str`, *optional*, defaults to `10GB`):
                Max model shard size to be used.
            safe_serializartion (`bool`, *optional*, defaults to `False`):
                Whether to save model using safe tensors or the traditional PyTorch way. If `True`, some tensors
                will be lost.
        """

        self.hps_config = hps_file_config
        self.checkpoint = checkpoint
        self.resume = resume
        self.model_path = model_path
        self.model_saving = model_saving.lower()
        self.enable_checkpointing = enable_checkpointing
        self.checkpoint_every = checkpoint_every
        self.logging_dir = logging_dir
        self.log_every = log_every
        self.grad_accumulation_steps = grad_accumulation_steps if grad_accumulation_steps else 1
        self.set_to_none = set_to_none
        self.shuffle_train = shuffle_train
        self.shuffle_validation = shuffle_validation
        self.model_saving_below_loss = model_saving_below_loss
        self.collate_fn = collate_fn
        self.max_shard_size = max_shard_size
        self.safe_serialization = safe_serialization

        self.accelerator = Accelerator(gradient_accumulation_steps=self.grad_accumulation_steps)
        self.accelerator.project_configuration = ProjectConfiguration(project_dir=".", logging_dir=logging_dir, total_limit=1)
        self.accelerator.log_with = [log_with]

    def fit(self,
            module: AcceleratorModule,
            train_dataset: Dataset,
            val_dataset: Dataset = None
    ):
        """
        Function to train a given `AcceleratorModule`.

        Args:
            module (`AcceleratorModule`):
                `AcceleratorModule` class containig the training logic.
            train_dataset (`torch.utils.data.Dataset`):
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

        if self.hps_config is None:
            raise AttributeError("Cannot train without HPS file config.")

        model = getattr(module, "model", None)
        if model is None:
            raise AttributeError("'self.model' needs to be declared in the AcceleratorModule class.")
        
        teacher = getattr(module, "teacher", None)
        
        cfg = read(self.hps_config)
        hps = cfg["hps"]
        optim = hps["optim"]
        schlr = hps["scheduler"] if "scheduler" in hps else None

        if self.model_path is None:
            self.model_path = cfg["version"]

        if self.model_saving:
            os.makedirs(self.model_path, exist_ok=True)

        best_train_loss = float("inf")
        best_valid_loss = float("inf")
        status_epoch = 0
        current_epoch_step = 0
        global_step = 0

        if self.resume:
            status = read_status(f"{self.checkpoint}/status.json")
            best_valid_loss = status["best_valid_loss"]
            best_train_loss = status["best_train_loss"]
            status_epoch = status["epoch"]
            global_step = status["global_step"]

        if module._implemented_collate_fn:
            self.collate_fn = module.collate_fn
        train_dataloader = DataLoader(train_dataset, batch_size=hps["batch_size"], shuffle=self.shuffle_train, collate_fn=self.collate_fn)

        val_dataloader = None
        if val_dataset is not None:
            val_dataloader = DataLoader(val_dataset, batch_size=hps["batch_size"], shuffle=self.shuffle_validation, collate_fn=self.collate_fn)
        else:
            if self.model_saving == "best_valid_loss":
                self.model_saving = "best_train_loss"

        optimizer = self._get_optimizer(optim, model)
        scheduler = None
        if schlr is not None:
            scheduler = self._get_scheduler(schlr, optimizer, -1, len(train_dataloader), hps["epochs"])
            # -1 for last_epoch since Accelerate will take care of recovering the progress

        model, train_dataloader, val_dataloader, optimizer, scheduler, teacher = self.accelerator.prepare(
            model, train_dataloader, val_dataloader, optimizer, scheduler, teacher
        )

        if scheduler:
            self.accelerator.register_for_checkpointing(scheduler)
        self.accelerator.init_trackers(self.model_path.split("/")[-1])

        if self.resume:
            if os.path.exists(self.checkpoint):
                self.accelerator.load_state(self.checkpoint)
            else:
                raise FileNotFoundError(f"{self.checkpoint} was not found.")

        epochs = hps["epochs"]
        eval_step = (len(train_dataloader) // len(val_dataloader)) if val_dataloader else None
        for epoch in range(status_epoch, epochs):
            eval_global_step = global_step
            model.train()
            train_losses = [] 
            for step, batch in tqdm(
                iterable=enumerate(train_dataloader, current_epoch_step+1),
                total=len(train_dataloader),
                desc=f"Epoch {epoch}/{epochs}",
                unit="batch"
            ):
                global_step, current_epoch_step = self._train_logic(
                    module, optimizer, batch, train_losses, step, scheduler, train_dataloader, global_step, current_epoch_step
                )
            
            if all([val_dataloader, getattr(module, "validation_step", False)]):
                model.eval()
                eval_losses = []
                with torch.no_grad():
                    for step, batch in tqdm(
                        iterable=enumerate(val_dataloader, 1),
                        total=len(val_dataloader),
                        desc=f"Epoch {epoch}/{epochs}",
                        unit="batch"
                    ):
                        eval_global_step = self._validation_logic(
                            module, batch, eval_losses, step, eval_step, eval_global_step
                        )

                best_valid_loss, best_train_loss = self._save_model_on_criteria(
                    model, eval_losses, train_losses, best_valid_loss, best_train_loss
                )
            else:
                if self.model_saving:
                    avg_train_loss = np.mean(train_losses)
                    if avg_train_loss < best_train_loss:
                        self.accelerator.print("Saving model...")
                        self._save_model(model, best_valid_loss, best_train_loss)
                        best_train_loss = avg_train_loss

            self._save_checkpoint(epoch+1, best_valid_loss, best_train_loss, global_step)

        self.accelerator.end_training()

    def eval(self, module, val_dataset: Dataset, batch_size=1) -> float:
        from torch.utils.data import DataLoader

        model = getattr(module, "model", None)
        if model is None:
            raise AttributeError("'self.model' needs to be declared in the AcceleratorModule class.")
        
        if module._implemented_collate_fn:
            self.collate_fn = module.collate_fn
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)

        model, val_dataloader = self.accelerator.prepare(
            model, val_dataloader
        )

        model.eval()
        eval_losses = []
        with torch.no_grad():
            for batch in tqdm(iterable=val_dataloader, total=len(val_dataloader), desc=f"Evaluating", unit="batch"):
                loss = module.validation_step(batch)
                loss = self.accelerator.gather_for_metrics(loss).cpu().numpy()
                eval_losses.append(np.mean(loss))
        
        avg_eval_loss = np.mean(eval_losses)

        return avg_eval_loss
    
    def _train_logic(
            self, module, optimizer, batch, train_losses, step, scheduler, dataloader, global_step, current_epoch_step
    ):
        loss = module.training_step(batch)
        if loss is None:
            loss = module.step(batch)

        if self.grad_accumulation_steps > 1:
            loss /= self.grad_accumulation_steps

        train_losses.append(loss.item())
        if step % self.log_every == 0 and self.accelerator.is_main_process:
            self.accelerator.log({"loss": {"train": loss}}, step=global_step)
        
        self.accelerator.backward(loss)

        if (step % self.grad_accumulation_steps == 0) or (step == len(dataloader)):
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=self.set_to_none)

        global_step += 1
        current_epoch_step += 1

        return global_step, current_epoch_step
    
    def _validation_logic(self, module, batch, eval_losses, step, eval_step, eval_global_step):
        loss = module.validation_step(batch)
        if loss is None:
            loss = module.step(batch)

        eval_losses.append(loss.item())
        if step % self.log_every == 0:
            self.accelerator.log({"loss": {"valid": loss.item()}}, step=eval_global_step)

        eval_global_step += eval_step

        return eval_global_step

    def _save_model(self, model, best_valid_loss, best_train_loss):
        self.accelerator.wait_for_everyone()
        state_dict = self.accelerator.get_state_dict(model)
        unwrapped_model = self.accelerator.unwrap_model(model)
        if getattr(unwrapped_model, "save_pretrained", None) is not None:
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
                unwrapped_model,
                f"{self.model_path}/pytorch_model.bin",
                safe_serialization=self.safe_serialization
            )

        save_status({
            "best_valid_loss": best_valid_loss,
            "best_train_loss": best_train_loss,
        }, to=f"{self.model_path}/status.json")

    
    def _save_model_on_criteria(self, model, eval_losses, train_losses, best_valid_loss, best_train_loss):
        if self.model_saving is None:
            return

        avg_valid_loss = np.mean(eval_losses)
        avg_train_loss = np.mean(train_losses)

        saving_criteria = {
            "best_valid_loss": avg_valid_loss < best_valid_loss and avg_valid_loss < self.model_saving_below_loss,
            "best_train_loss": avg_train_loss < best_train_loss and avg_train_loss < self.model_saving_below_loss,
            "always": True
        }

        if self.model_saving in saving_criteria:
            if saving_criteria[self.model_saving]:
                self._save_model(model, best_valid_loss, best_train_loss)
        else:
            raise ValueError("Invalid type of model saving. Value must be: "
                              "'best_valid_train_loss', "
                              "'best_train_loss', or "
                              "'always'.")
        
        return (
            avg_valid_loss if avg_valid_loss < best_valid_loss else best_valid_loss,
            avg_train_loss if avg_train_loss < best_train_loss else best_train_loss
        )
    
    def _fix_kwargs(self, dictionary: dict):
        for k, v in dictionary.items():
            if isinstance(v, list):
                dictionary[k] = [float(item) for item in v if isinstance(item, str)]
            elif isinstance(v, str):
                try:
                    dictionary[k] = float(v)
                except ValueError:
                    continue

    def _save_checkpoint(self, epoch, best_valid_loss, best_train_loss, global_step):
        if (self.enable_checkpointing and epoch % self.checkpoint_every == 0):
            self.accelerator.print("Saving checkpoint...")
            self.accelerator.save_state(self.checkpoint, safe_serialization=self.safe_serialization)
            save_status({
                "best_valid_loss": best_valid_loss,
                "best_train_loss": best_train_loss,
                "epoch": epoch,
                "global_step": global_step
            }, to=f"{self.checkpoint}/status.json")

    def _get_optimizer(self, optim: dict, model):
        t = optim["type"]
        optim_kwargs = optim.copy()
        del optim_kwargs["type"]
        self._fix_kwargs(optim_kwargs)

        return OPTIMIZERS[t](model.parameters(), **optim_kwargs)

    def _filter_kwargs(self, _kwargs: dict, fn):
        return {k:v for k,v in _kwargs.items() if k in fn.__init__.__code__.co_varnames}

    def _get_scheduler(self, schlr: dict, optimizer, last_epoch, steps_per_epoch, epochs):
        t = schlr["type"]
        schlr_kwargs = schlr.copy()
        del schlr_kwargs["type"]
        self._fix_kwargs(schlr_kwargs)

        schlr_kwargs["last_epoch"] = last_epoch
        schlr_kwargs["steps_per_epoch"] = steps_per_epoch
        schlr_kwargs["epochs"] = epochs
        filtered_kwargs = self._filter_kwargs(schlr_kwargs, SCHEDULERS[t])

        return SCHEDULERS[t](optimizer, **filtered_kwargs)
