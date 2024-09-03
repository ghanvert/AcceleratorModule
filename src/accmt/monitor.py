import os
import psutil
import torch
from accelerate import Accelerator, DistributedType
from dataclasses import dataclass
from typing_extensions import Union

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
        accuracy (`bool`, *optional*, defaults to `True`):
            Monitor accuracy if implemented.
        grad_norm (`bool`, *optional*, defaults to `False`):
            This will enable monitoring for gradient normalization. This feature is not yet supported 
            when running with DeepSpeed.
        gpu_utilization (`bool`, *optional*, defaults to `False`):
            Monitor GPU utilization in GB. It only reports GPU from main process (for now).
        cpu_utilization (`bool`, *optional*, defaults to `False`):
            Monitor CPU utilization in GB. It only reports CPU from main process (for now)
        val_equal_train (`bool`, *optional*, defaults to `True`):
            When reporting validation loss and accuracy, its step will be equal to train loss. If set to 
            `False`, validation step will be equal to the number of evaluations done (starting at 0). 
            This argument is only valid when `report_loss_after_eval` is set to `True`.
    """
    def __init__(self,
                 learning_rate: bool = False,
                 train_loss: bool = True,
                 validation_loss: bool = True,
                 additional_metrics: bool = True,
                 grad_norm: bool = False,
                 gpu_utilization: bool = False,
                 cpu_utilization: bool = False,
                 val_equal_train: bool = True
    ):
        self.learning_rate = learning_rate
        self.train_loss = train_loss
        self.validation_loss = validation_loss
        self.additional_metrics = additional_metrics
        self.grad_norm = grad_norm
        self.gpu_utilization = gpu_utilization
        self.cpu_utilization = cpu_utilization
        self.val_equal_train = val_equal_train
        self.status_dict = None
        self.accelerator = None
        self.train_loss_name = None
        self.validation_loss_name = None
        self._do_tracking = True

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
        if self.learning_rate and self.accelerator.is_main_process and self._do_tracking:
            self.accelerator.log({"learning_rate": self.status_dict["learning_rate"]}, step=self.status_dict["global_step"]+1)

    def log_train_loss(self):
        if self.train_loss and self.accelerator.is_main_process and self._do_tracking:
            self.accelerator.log({self.train_loss_name: self.status_dict["train_loss"]}, step=self.status_dict["global_step"]+1)

    def log_validation_loss(self):
        if self.validation_loss and self.accelerator.is_main_process and self._do_tracking:
            step = self.status_dict["eval_global_step"] if self.val_equal_train else self.status_dict["evaluations_done"]
            self.accelerator.log({self.validation_loss_name: self.status_dict["validation_loss"]}, step=step)

    def log_additional_metrics(self):
        if self.additional_metrics and self.accelerator.is_main_process and self._do_tracking:
            step = self.status_dict["test_global_step"] if self.val_equal_train else self.status_dict["evaluations_done"]
            for metric, value in self.status_dict["additional_metrics"].items():
                self.accelerator.log({metric: value}, step=step)

    def log_gpu_utilization(self):
        if self.gpu_utilization and self.accelerator.is_main_process and self._do_tracking:
            if self.accelerator.distributed_type in {
                DistributedType.DEEPSPEED,
                DistributedType.FSDP,
                DistributedType.MULTI_GPU
            }:
                device = torch.device("cuda")
                memory_allocated = torch.cuda.memory_allocated(device)
                memory_reserved = torch.cuda.memory_reserved(device)
                total_memory = (memory_allocated + memory_reserved) / (1024**3)
            
                self.accelerator.log({"GPU_0": total_memory}, step=self.status_dict["global_step"]+1)

    def log_cpu_utilization(self):
        if self.cpu_utilization and self.accelerator.is_main_process and self._do_tracking:
            process = psutil.Process(os.getpid())
            cpu_mem = process.memory_info().rss / (1024**3)
            self.accelerator.log({"CPU_PROCESS_0": cpu_mem}, step=self.status_dict["global_step"]+1)

    def log_grad_norm(self):
        if self.grad_norm and self.accelerator.is_main_process and "grad_norm" in self.status_dict and self._do_tracking:
            self.accelerator.log({"grad_norm": self.status_dict["grad_norm"]}, step=self.status_dict["global_step"]+1)
