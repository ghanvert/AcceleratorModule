import torch
from accelerate import Accelerator, InitProcessGroupKwargs, DataLoaderConfiguration
from datetime import timedelta

def allow_tf32(flag=True):
    """Enable or disable the use of TensorFloat32."""
    torch.set_float32_matmul_precision("high" if flag else "highest")

allow_tf32()

_init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=86400))
_dataloader_config = DataLoaderConfiguration(use_seedable_sampler=True)
accelerator = Accelerator(kwargs_handlers=[_init_kwargs], dataloader_config=_dataloader_config, step_scheduler_with_optimizer=False)

from .trainer import (
    AcceleratorModule,
    Trainer,
    set_seed
)
from .collate_fns import (
    DataCollatorForSeq2Seq,
    DataCollatorForLanguageModeling,
    DataCollatorForLongestSequence
)
from .tracker import (
    TensorBoard,
    WandB,
    CometML,
    Aim,
    MLFlow,
    ClearML,
    DVCLive
)
from .handlers import Handler
from .dataloader_samplers import TemperatureSampler
from .monitor import Monitor
from .hyperparameters import HyperParameters, Optimizer, Scheduler
from .utility import prepare, prepare_array, prepare_dataframe, prepare_list
from accelerate.utils import tqdm
