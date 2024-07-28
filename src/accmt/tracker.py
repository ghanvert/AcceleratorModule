from dataclasses import dataclass
from accelerate.utils import LoggerType

@dataclass
class TensorBoard:
    tracker = LoggerType.TENSORBOARD
    init = lambda **kwargs: {"tensorboard": {**kwargs}}

@dataclass
class WandB:
    tracker = LoggerType.WANDB
    init = lambda **kwargs: {"wandb": {**kwargs}}

@dataclass
class CometML:
    tracker = LoggerType.COMETML
    init = lambda **kwargs: {"comet_ml": {**kwargs}}

@dataclass
class Aim:
    tracker = LoggerType.AIM
    init = lambda **kwargs: {"aim": {**kwargs}}

@dataclass
class MLFlow:
    tracker = LoggerType.MLFLOW
    init = lambda **kwargs: {"mlflow": {**kwargs}}

@dataclass
class ClearML:
    tracker = LoggerType.CLEARML
    init = lambda **kwargs: {"clearml": {**kwargs}}

@dataclass
class DVCLive:
    tracker = LoggerType.DVCLIVE
    init = lambda **kwargs: {"dvclive": {**kwargs}}
