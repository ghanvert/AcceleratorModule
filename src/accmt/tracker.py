from accelerate.utils import LoggerType

class TensorBoard: tracker = LoggerType.TENSORBOARD
class WandB: tracker = LoggerType.WANDB
class CometML: tracker = LoggerType.COMETML
class Aim: tracker = LoggerType.AIM
class MLFlow: tracker = LoggerType.MLFLOW
class ClearML: tracker = LoggerType.CLEARML
class DVCLive: tracker = LoggerType.DVCLIVE
