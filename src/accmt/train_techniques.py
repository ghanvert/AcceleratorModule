import torch.nn as nn
import random
from torch.nn.utils import prune

class RandomPruningInModules:
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
    def __init__(self,
                 model: nn.Module,
                 amount=0.1,
                 pruning_method=prune.L1Unstructured,
                 importance_scores=None
    ):
        self.amount = amount
        self.pruning_method = pruning_method
        self.importance_scores = importance_scores

        self.modules = self._get_random_modules(model, self.amount)

    def __call__(self):
        prune.global_unstructured(
            self.modules,
            pruning_method=self.pruning_method,
            importance_scores=self.importance_scores,
            amount=self.amount
        )

    def _get_random_modules(self, model: nn.Module, amount) -> list:
        modules = list(model.children())
        num_modules = len(modules)
        num_selected = int(num_modules * amount)

        return random.sample(modules, num_selected)
