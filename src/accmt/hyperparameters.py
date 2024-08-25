import torch
import yaml
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
    def __init__(self,
                 epochs: int = 1,
                 batch_size: Union[int, tuple[int]] = 1,
                 optim: Union[str, Optimizer] = "SGD",
                 optim_kwargs: Optional[dict] = None,
                 scheduler: Optional[Union[str, Scheduler]] = None,
                 scheduler_kwargs: Optional[dict] = None
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.optim = getattr(Optimizer, optim) if isinstance(optim, str) else optim
        self._fix_kwargs(optim_kwargs)
        self.optim_kwargs = optim_kwargs if optim_kwargs is not None else {}
        self.scheduler = getattr(Scheduler, scheduler) if isinstance(scheduler, str) else scheduler
        self._fix_kwargs(scheduler_kwargs)
        self.scheduler_kwargs = scheduler_kwargs if scheduler_kwargs is not None else {}

    @classmethod
    def from_config(cls, config: Union[str, dict]):
        if isinstance(config, str):
            config = yaml.safe_load(open(config))["hps"]
        elif "hps" in config:
            config = config["hps"]
        optimizer = config["optim"]
        scheduler = config["scheduler"] if "scheduler" in config else None

        return HyperParameters(
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            optim=optimizer["type"],
            optim_kwargs={k:v for k,v in optimizer.items() if k != "type"} if len(optimizer) > 1 else None,
            scheduler=scheduler,
            scheduler_kwargs={k:v for k,v in scheduler.items() if k != "type"} if scheduler is not None and len(scheduler) > 1 else None
        )

    def to_dict(self) -> dict:
        optim = self.optim if not isinstance(self.optim, str) else getattr(Optimizer, self.optim, None)
        assert optim is not None, f"{optim} is not a valid optimizer."
        scheduler = self.scheduler if not isinstance(self.scheduler, str) else getattr(Scheduler, self.scheduler, "INVALID")
        assert scheduler != "INVALID", f"{scheduler} is not a valid scheduler."

        optim_kwargs = self.optim_kwargs if self.optim_kwargs is not None else {}
        schlr_kwargs = self.scheduler_kwargs if self.scheduler_kwargs is not None else {}

        d = {
            "hps": {
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "optim": {"type": optim, **optim_kwargs}
            }
        }

        if self.scheduler is not None:
            d["hps"]["scheduler"] = {"type": scheduler, **schlr_kwargs}

        return d

    def get_config(self) -> dict:
        hps = self.to_dict()["hps"]
        _hps = {"epochs": hps["epochs"], "batch_size": hps["batch_size"], **hps["optim"]}
        if "type" in _hps:
            t = _hps["type"]
            _hps["optimizer"] = t if isinstance(t, str) else t.__name__
            del _hps["type"]

        if "scheduler" in hps:
            t = hps["scheduler"]["type"]
            for k, v in hps["scheduler"].items():
                if k == "type":
                    _hps["scheduler"] = v if isinstance(t, str) else t.__name__
                    continue
                _hps[k] = v

        return _hps

    def __getitem__(self, key: str):
        return getattr(self, key)
    
    def _fix_kwargs(self, dictionary: Optional[dict]):
        if dictionary is None: return
        for k, v in dictionary.items():
            if isinstance(v, str):
                try:
                    dictionary[k] = float(v)
                except ValueError:
                    continue
