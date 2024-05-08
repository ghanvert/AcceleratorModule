import torch.nn as nn
import random
from torch.nn.utils import prune
from .events import *

class RandomPruningInModules(EpochStart):
    """
    Makes a random amount of pruning in the specified modules.
    """
    def __init__(self,
                 modules,
                 amount=0.1,
                 pruning_method=prune.L1Unstructured,
                 importance_scores=None
    ):
        self.modules = modules
        self.amount = amount
        self.pruning_method = pruning_method
        self.importance_scores = importance_scores

    def __call__(self):
        prune.global_unstructured(
            self.modules,
            pruning_method=self.pruning_method,
            importance_scores=self.importance_scores,
            amount=self.amount
        )

class RandomPruning(EpochStart):
    """
    Makes a random amount of pruning in some random modules.
    """
    def __init__(self,
                 model: nn.Module,
                 amount=0.1,
                 pruning_method=prune.L1Unstructured,
                 importance_scores=None
    ):
        self.model = model
        self.amount = amount
        self.pruning_method = pruning_method
        self.importance_scores = importance_scores

        self.modules = self._get_random_modules()

    def __call__(self):
        prune.global_unstructured(
            self.modules,
            pruning_method=self.pruning_method,
            importance_scores=self.importance_scores,
            amount=self.amount
        )

    def _get_random_modules(self) -> list:
        modules = list(self.model.children())
        num_modules = len(modules)
        num_selected = int(num_modules * self.amount)

        return random.sample(modules, num_selected)

class LabelSmoothing(OnBatch):
    """
    Applies label smoothing regulatization technique for one-hot target tensors.
    """
    def __init__(self, smoothing=0.1, key: str = None):
        """
        Args:
            smoothing (`float`, *optional*, defaults to `0.1`):
                Smoothing or alpha value to smooth one hot vectors.
            key (`str`, *optional*, defaults to `None`):
                In case that every batch is a dictionary, this will be the labels key.
        """
        self.smoothing = smoothing
        self.key = key

    def __call__(self, batch):
        if isinstance(batch, dict):
            assert self.key in batch.keys(), f"'{self.key}' is not a valid key."
            target = batch[self.key]
        else:
            _, target = batch
        
        if getattr(target, "shape", None) is not None:
            K = target.shape[-1]
        else:
            K = len(target)
        
        return (1 - self.smoothing) * target + self.smoothing / K
    
class GradientNormClipping(AfterBackward):
    """
    Applies gradient normalization clipping.
    """
    def __init__(self,
                 max_norm=1.0,
                 norm_type=2.0,
                 error_if_nonfinite=False,
                 foreach=None
    ):
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.error_if_nonfinite = error_if_nonfinite
        self.foreach = foreach

    def __call__(self, parameters):
        nn.utils.clip_grad.clip_grad_norm_(
            parameters,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            error_if_nonfinite=self.error_if_nonfinite,
            foreach=self.foreach
        )

class GradientValueClipping(AfterBackward):
    """
    Applies gradient clipping to a specific value.
    """
    def __init__(self, clip_value, foreach=None):
        self.clip_value = clip_value
        self.foreach = foreach

    def __call__(self, parameters):
        nn.utils.clip_grad.clip_grad_value_(
            parameters,
            clip_value=self.clip_value,
            foreach=self.foreach
        )

class RandomFreezing(EpochEnd):
    """
    Freezes random parameters at the end of every epoch.
    """
    def __init__(self, parameters, amount=0.1):
        self.parameters = list(parameters)
        self.amount = amount
        self.num_parameters_to_select = int(len(self.parameters) * amount)
        self.frozen = []

    def __call__(self):
        self.unfreeze()
        self.choose_parameters()
        self.freeze()

    def choose_parameters(self):
        self.frozen = random.sample(self.parameters, self.num_parameters_to_select)
    
    def freeze(self):
        for param in self.frozen:
            param.requires_grad = False

    def unfreeze(self):
        for param in self.frozen:
            param.requires_grad = True

class EternalFreeze(Start, RandomFreezing):
    """
    Freeze a random amount of parameters during all training.
    """
    def __init__(self, parameters, amount=0.1):
        super().__init__(parameters, amount)

    def __call__(self):
        self.choose_parameters()
        self.freeze()
