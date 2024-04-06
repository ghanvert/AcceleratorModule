import numpy as np
import torch
import warnings

from abc import ABC
from accelerate import Accelerator
from accelerate.utils import LoggerType, ProjectConfiguration
from .config import read, save_status, read_status
from torch.utils.data import Dataset
from typing import Any
from typing_extensions import override


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
        `training_step`:
            Defines the training logic. Must return a loss `torch.Tensor` (scalar).
        `validation_step` (optional):
            Defines the validation logic. Must return a loss `torch.Tensor` (scalar).
            If not implemented, no validation will be executed.
    
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

    @override
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Defines the flow of data."""
    
    @override
    def training_step(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Defines the training logic. Must return a loss tensor (scalar)."""
    
    @override
    def validation_step(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Defines the validation logic. Must return a loss tensor (scalar)."""
    
    def __init_subclass__(cls, **kwargs):
        if cls.training_step == AcceleratorModule.training_step and cls.validation_step == AcceleratorModule.validation_step:
            raise TypeError(
                "Subclasses of 'Trainer' must override 'training_step' and/or 'validation_step' methods."
            )
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
                hps_file_config: str,
                checkpoint = "checkpoint1",
                resume = False,
                model_path: str = None,
                model_saving = "best_valid_loss",
                enable_checkpointing = True,
                checkpoint_every = 1,
                logging_dir = "logs",
                log_with = LoggerType.TENSORBOARD,
                log_every = 1,
                grad_accumulation_steps=1,
                set_to_none=True
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
            grad_accumulation_steps (`int`, *optional*, defaults to `1`):
                Accumulate gradients for N steps. Useful for training large models and simulate
                large batches when memory is not enough. If set to `1`, no accumulation will be perfomed.
            set_to_none (`bool`, *optional*, defaults to `True`):
                From PyTorch documentation: "instead of setting to zero, set the grads to None. This will
                in general have lower memory footprint, and can modestly improve performance." Some
                optimizers have a different behaviour if the gradient is 0 or None. See PyTorch docs
                for more information: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
        """

        self.accelerator = Accelerator()
        self.hps_config = hps_file_config
        self.checkpoint = checkpoint
        self.resume = resume
        self.model_path = model_path
        self.model_saving = model_saving.lower()
        self.enable_checkpointing = enable_checkpointing
        self.checkpoint_every = checkpoint_every
        self.logging_dir = logging_dir
        self.log_every = log_every
        self.grad_accumulation_steps = grad_accumulation_steps
        self.set_to_none = set_to_none

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

        from tqdm.auto import tqdm
        from torch.utils.data import DataLoader

        model = getattr(module, "model", None)
        
        if not model:
            raise AttributeError("'self.model' needs to be declared in the AcceleratorModule class.")
        
        cfg = read(self.hps_config)
        hps = cfg["hps"]
        optim = hps["optim"]
        schlr = hps["scheduler"]

        if not self.model_path:
            self.model_path = cfg["version"]

        if self.model_saving:
            os.makedirs(self.model_path, exist_ok=True)

        train_dataloader = DataLoader(train_dataset, batch_size=hps["batch_size"], shuffle=True)

        val_dataloader = None
        if val_dataset is not None:
            val_dataloader = DataLoader(val_dataset, batch_size=hps["batch_size"], shuffle=True)
        else:
            if self.model_saving == "best_valid_loss":
                self.model_saving = "best_train_loss"
        
        optimizer = getattr(torch.optim, optim["type"])(model.parameters(), lr=float(optim["lr"]), weight_decay=float(optim["weight_decay"]))
        scheduler = None
        if "type" in schlr:
            scheduler = getattr(torch.optim.lr_scheduler, schlr["type"])(optimizer, max_lr=float(schlr["max_lr"]), steps_per_epoch=len(train_dataloader), epochs=hps["epochs"])

        if "log_every" in cfg:
            self.log_every = cfg["log_every"]
            warnings.warn("'log_every' parameter in HPS config file is deprecated and it'll be removed in "
                          "v1.0.0. Use 'log_every' in Trainer constructor instead.\n"
                          "Using 'log_every' from HPS config file.")

        model, train_dataloader, val_dataloader, optimizer, scheduler = self.accelerator.prepare(
            model, train_dataloader, val_dataloader, optimizer, scheduler
        )
        self.accelerator.init_trackers(cfg["version"])

        best_valid_loss = float("inf")
        best_train_loss = float("inf")
        status_epoch = 0

        if self.resume:
            if os.path.exists(self.checkpoint):
                self.accelerator.load_state(self.checkpoint)
                status = read_status(f"{self.checkpoint}/status.json")
                best_valid_loss = status["best_valid_loss"]
                best_train_loss = status["best_train_loss"]
                status_epoch = status["epoch"]
            else:
                print(f"{self.checkpoint} does not exist. Starting process from zero...")

        epochs = hps["epochs"]
        global_step = 0
        eval_step = len(train_dataloader) // len(val_dataloader)
        for epoch in range(status_epoch, epochs):
            eval_global_step = global_step
            model.train()
            train_losses = []
            for step, batch in tqdm(enumerate(train_dataloader, 1), total=len(train_dataloader), desc=f"Epoch {epoch}/{epochs}", unit="batch"):
                loss = module.training_step(batch)

                if self.grad_accumulation_steps > 1:
                    loss /= self.grad_accumulation_steps

                train_losses.append(loss.item())
                if step % self.log_every == 0:
                    self.accelerator.log({"loss": {"train": loss.item()}}, step=global_step)

                self.accelerator.backward(loss)

                if (step % self.grad_accumulation_steps == 0) or (step == len(train_dataloader)):
                    optimizer.step()
                    if scheduler:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=self.set_to_none)

                global_step += 1
            
            if all([val_dataloader, getattr(module, "validation_step", False)]):
                model.eval()
                eval_losses = []
                with torch.no_grad():
                    for step, batch in tqdm(enumerate(val_dataloader, 1), total=len(val_dataloader), desc=f"Epoch {epoch}/{epochs}", unit="batch"):
                        loss = module.validation_step(batch)

                        eval_losses.append(loss.item())
                        if step % self.log_every == 0:
                            self.accelerator.log({"loss": {"valid": loss.item()}}, step=eval_global_step)

                        eval_global_step += eval_step

                best_valid_loss, best_train_loss = self._save_model_on_criteria(
                    model, eval_losses, train_losses, best_valid_loss, best_train_loss
                )
            else:
                if self.model_saving:
                    avg_train_loss = np.mean(train_losses)
                    if avg_train_loss < best_train_loss:
                        self._save_model(model, best_valid_loss, best_train_loss)
                        best_train_loss = avg_train_loss

            if self.enable_checkpointing and epoch % self.checkpoint_every == 0:
                self.accelerator.save_state(self.checkpoint)
                save_status({
                    "best_valid_loss": best_valid_loss,
                    "best_train_loss": best_train_loss,
                    "epoch": epoch
                }, to=f"{self.checkpoint}/status.json")

        self.accelerator.end_training()

    
    def _save_model(self, model, best_valid_loss, best_train_loss):
        state_dict = self.accelerator.get_state_dict(model)
        unwrapped_model = self.accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            self.model_path,
            is_main_process=self.accelerator.is_main_process,
            state_dict=state_dict,
            max_shard_size="10GB",
            save_function=self.accelerator.save,
            safe_serialization=False # if True, some tensors will not be saved
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
            "best_valid_loss": (avg_valid_loss < best_valid_loss),
            "best_train_loss": (avg_train_loss < best_train_loss),
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