import torch
from torch.optim import lr_scheduler
from dataclasses import dataclass
from typing_extensions import Optional, Union
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

@dataclass
class Optimizer:
    Adam = torch.optim.Adam
    Adadelta = torch.optim.Adadelta
    Adagrad = torch.optim.Adagrad
    Adamax = torch.optim.Adamax
    AdamW = torch.optim.AdamW
    Adafactor = Adafactor
    ASGD = torch.optim.ASGD
    NAdam = torch.optim.NAdam
    RAdam = torch.optim.RAdam
    RMSprop = torch.optim.RMSprop
    Rprop = torch.optim.Rprop
    SGD = torch.optim.SGD
    SparseAdam = torch.optim.SparseAdam

@dataclass
class Scheduler:
    StepLR = lr_scheduler.StepLR
    LinearLR = lr_scheduler.LinearLR
    ExponentialLR = lr_scheduler.ExponentialLR
    CosineAnnealingLR = lr_scheduler.CosineAnnealingLR
    CyclicLR = lr_scheduler.CyclicLR
    OneCycleLR = lr_scheduler.OneCycleLR
    CosineAnnealingWarmRestarts = lr_scheduler.CosineAnnealingWarmRestarts
    CosineWithWarmup = get_cosine_schedule_with_warmup
    Constant = get_constant_schedule
    ConstantWithWarmup = get_constant_schedule_with_warmup
    CosineWithHardRestartsWithWarmup = get_cosine_with_hard_restarts_schedule_with_warmup
    InverseSQRT = get_inverse_sqrt_schedule
    LinearWithWarmup = get_linear_schedule_with_warmup
    PolynomialDecayWithWarmup = get_polynomial_decay_schedule_with_warmup

@dataclass
class HyperParameters:
    """
    Class to set hyperparameters for training.

    Args:
        epochs (`int`, *optional*, defaults to `1`):
            Number of epochs (how many times we run the model over the dataset).
        batch_size (`int` or `tuple`, *optional*, defaults to `1`):
            Batch size (how many samples are passed to the model at the same time). This can also be a 
            `tuple`, the first element indicating batch size during training, and the second element 
            indicating batch size during evaluation.

            NOTE: This is not effective batch size. Effective batch size will be calculated multiplicating 
            this value by the number of processes.
        optim (`str` or `Optimizer`, *optional*, defaults to `SGD`):
            Optimization algorithm. See documentation to check the available ones.
        optim_kwargs (`dict`, *optional*, defaults to `None`):
            Specific optimizer keyword arguments.
        scheduler (`str` or `Scheduler`, *optional*, defaults to `None`):
            Learning rate scheduler to implement.
        scheduler_kwargs (`dict`, *optional*, defaults to `None`):
            Specific scheduler keyword arguments.
    """
    epochs: int = 1
    batch_size: Union[int, tuple[int]] = 1
    optim: Union[str, Optimizer] = "SGD"
    optim_kwargs: Optional[dict] = {}
    scheduler: Optional[Union[str, Scheduler]] = None
    scheduler_kwargs: Optional[dict] = {}

    def to_dict(self) -> dict:
        optim = self.optim if not isinstance(self.optim, str) else getattr(Optimizer, self.optim, None)
        assert optim is not None, f"{optim} is not a valid optimizer."
        scheduler = self.scheduler if not isinstance(self.scheduler, str) else getattr(Scheduler, self.scheduler, "INVALID")
        assert scheduler != "INVALID", f"{scheduler} is not a valid scheduler."

        d = {
            "hps": {
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "optim": {"type": self.optim, **self.optim_kwargs}
            }
        }

        if isinstance(self.scheduler, str):
            d["hps"]["scheduler"] = {"type": self.scheduler, **self.scheduler_kwargs}

        return d

