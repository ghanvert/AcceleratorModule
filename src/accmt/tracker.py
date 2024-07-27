from dataclasses import dataclass
from accelerate.utils import LoggerType

@dataclass
class TensorBoard:
    tracker = LoggerType.TENSORBOARD
    init = lambda name, **kwargs: {"tensorboard": {"run_name": name, **kwargs}}

@dataclass
class WandB:
    tracker = LoggerType.WANDB
    init = lambda name, **kwargs: {"wandb": {"run_name": name, **kwargs}}

@dataclass
class CometML:
    tracker = LoggerType.COMETML
    init = lambda name, **kwargs: {"comet_ml": {"run_name": name, **kwargs}}

@dataclass
class Aim:
    tracker = LoggerType.AIM
    init = lambda name, **kwargs: {"aim": {"run_name": name, **kwargs}}

@dataclass
class MLFlow:
    tracker = LoggerType.MLFLOW
    init = lambda name, **kwargs: {"mlflow": {"run_name": name, **kwargs}}

@dataclass
class ClearML:
    tracker = LoggerType.CLEARML
    init = lambda name, **kwargs: {"clearml": {"run_name": name, **kwargs}}

@dataclass
class DVCLive:
    tracker = LoggerType.DVCLIVE
    init = lambda name, **kwargs: {"dvclive": {"run_name": name, **kwargs}}
