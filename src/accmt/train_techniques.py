import torch.nn as nn
import random
from torch.nn.utils import prune

class RandomPruningInModules:
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

class RandomPruning:
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

class LabelSmoothing:
    """
    Applies label smoothing regulatization technique for one-hot target tensors.
    """
    def __init__(self, smoothing=0.1):
        self.smoothing = smoothing

    def __call__(self, x):
        if getattr(x, "shape", None) is not None:
            K = x.shape[-1]
        else:
            K = len(x)
        
        return (1 - self.smoothing) * x + self.smoothing / K
    
class GradientNormClipping:
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

class GradientValueClipping:
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

class RandomFreezing:
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

class EternalFreeze(RandomFreezing):
    """
    Freeze a random amount of parameters during all training.
    """
    def __init__(self, parameters, amount=0.1):
        super().__init__(parameters, amount)

    def __call__(self):
        self.choose_parameters()
        self.freeze()
