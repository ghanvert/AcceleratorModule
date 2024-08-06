import os
import psutil
import torch
from accelerate import Accelerator, DistributedType
from dataclasses import dataclass
from typing_extensions import Union, Optional

@dataclass
class Monitor:
    """
    Class to set metrics to monitor during training using a tracker (if implemented).

    Args:
        learning_rate (`bool`, *optional*, defaults to `False`):
            Monitor learning rate.
        train_loss (`bool`, *optional*, defaults to `True`):
            Monitor training loss.
        validation_loss (`bool`, *optional*, defaults to `True`):
            Monitor validation loss.
        gpu_utilization (`bool`, *optional*, defaults to `False`):
            Monitor GPU utilization.
        cpu_utilization (`bool`, *optional*, defaults to `False`):
            Monitor CPU utilization.
    """
    def __init__(self,
                 learning_rate: Optional[bool] = False,
                 train_loss: Optional[bool] = True,
                 validation_loss: Optional[bool] = True,
                 gpu_utilization: Optional[bool] = False,
                 cpu_utilization: Optional[bool] = False
    ):
        self.learning_rate = learning_rate
        self.train_loss = train_loss
        self.validation_loss = validation_loss
        self.gpu_utilization = gpu_utilization
        self.cpu_utilization = cpu_utilization
        self.status_dict = None
        self.accelerator = None
        self.train_loss_name = None
        self.validation_loss_name = None

    @classmethod
    def from_config(cls, config: Union[str, dict]):
        """
        Load a monitor configuration from a file or a dictionary.

        Args:
            config (`str` or `dict`):
                Path to a file or dictionary containing kwargs for Monitor constructor. The file can 
                be YAML or JSON.
        """
        assert config is None or isinstance(config, (str, dict)), f"{config} is not of type 'str' or 'dict'."
        if isinstance(config, str):
            import yaml
            config = yaml.safe_load(open(config))
        elif config is None:
            config = {}

        return Monitor(**config)
    
    def _set_extra(self, accelerator: Accelerator, status_dict: dict, train_loss_name: str, validation_loss_name: str):
        self.accelerator = accelerator
        self.status_dict = status_dict
        self.train_loss_name = train_loss_name
        self.validation_loss_name = validation_loss_name

    def log_learning_rate(self):
        if self.learning_rate and self.accelerator.is_main_process:
            self.accelerator.log({"learning_rate": self.status_dict["learning_rate"]}, step=self.status_dict["global_step"]+1)

    def log_train_loss(self):
        if self.train_loss and self.accelerator.is_main_process:
            self.accelerator.log({self.train_loss_name: self.status_dict["train_loss"]}, step=self.status_dict["global_step"]+1)

    def log_validation_loss(self):
        if self.validation_loss and self.accelerator.is_main_process:
            self.accelerator.log({self.validation_loss_name: self.status_dict["validation_loss"]}, step=self.status_dict["eval_global_step"]+1)

    def log_gpu_utilization(self):
        if self.gpu_utilization and self.accelerator.is_main_process:
            if self.accelerator.distributed_type in {
                DistributedType.DEEPSPEED,
                DistributedType.FSDP,
                DistributedType.MULTI_GPU
            }:
                num_processes = self.accelerator.num_processes
                gpu_dict = {}
                for process_idx in range(num_processes):
                    device = torch.device(f"cuda:{process_idx}")
                    memory_allocated = torch.cuda.memory_allocated(device)
                    memory_reserved = torch.cuda.memory_reserved(device)
                    total_memory = (memory_allocated + memory_reserved) / (1024**3)
                    gpu_dict[f"GPU_{process_idx}"] = total_memory
                    
                self.accelerator.log(gpu_dict, step=self.status_dict["global_step"]+1)

    def log_cpu_utilization(self):
        if self.cpu_utilization:
            num_processes = self.accelerator.num_processes
            for process_idx in range(num_processes):
                if self.accelerator.process_index == process_idx:
                    process = psutil.Process(os.getpid())
                    cpu_mem = process.memory_info().rss / (1024**3)
                    self.accelerator.log({f"CPU_PROCESS_{process_idx}": cpu_mem}, step=self.status_dict["global_step"]+1)