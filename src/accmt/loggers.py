from accelerate.utils import LoggerType

class TensorBoard: logger = LoggerType.TENSORBOARD
class WandB: logger = LoggerType.WANDB
class CometML: logger = LoggerType.COMETML
class Aim: logger = LoggerType.AIM
class MLFlow: logger = LoggerType.MLFLOW
class ClearML: logger = LoggerType.CLEARML
class DVCLive: logger = LoggerType.DVCLIVE
