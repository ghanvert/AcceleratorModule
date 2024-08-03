import yaml
import torch
import numpy as np
from torch.utils.data import Dataset, WeightedRandomSampler
from typing_extensions import Optional, Union, Callable
from .utils import PANDAS_READER_MAP

class TemperatureSampler:
    def __init__(self,
                 dataset: Dataset,
                 temperature: int,
                 distribution: Optional[Union[dict, str]] = None,
                 target_format: Optional[Callable] = None,
                 seed: Optional[int] = None
    ):
        """
        Temperature sampler based on the up-sampling technique described in 
        "Massively Multilingual Neural Machine Translation in the Wild: Findings 
        and Challenges" (https://arxiv.org/pdf/1907.05019).

        Useful for imbalanced datasets to up-sample low-resource classes.

        Args:
            dataset (`Dataset`):
                PyTorch Dataset object.
            temperature (`int`):
                Temperature for sampling. A value of `1` corresponds to true data 
                distribution and a value of `100` corresponds to (almost) equal number 
                of samples for each class (close to uniform distribution with over-sampled) 
                low-resource classes.
            distribution (`dict` or `str`, *optional*, defaults to `None`):
                Class distribution `dict` (with keys being the classes and values being 
                the number of samples of that class in the dataset) or `str` indicating a path to a 
                file containing a dict-like or Pandas DataFrame format (in this case, first column 
                must represent the class and the second column must represent the number of samples of 
                that class).

                If not specified, dataset distribution will be automatically calculed (this can be a 
                bottleneck when initializating).
            target_format (`tuple`, *optional*, defaults to `None`):
                Target format when iterating over the dataset with `__getitem__`. This indicates the format 
                on how to get access to every sample's label in order to calculate the corresponding 
                sample weight (probability).

                Examples:
                    target_format = lambda batch: batch[1] # indicates that the second element of the `__getitem__` return 
                    value is the class target.

                    target_format = lambda batch: batch[0]['language'] # indicates that the first element of the `__getitem__` 
                    return value is a dictionary containing the key 'language', which is the class target.

                If not specified, the target format will be inferred based on the first element of the dataset. It will get 
                the return type of `__getitem__`, and in the case it's a `tuple` or `list`, the last element of it will be 
                considered as the target. If the return type is a `dict`, it will look for keys 'label' or 'target'. If these 
                keys are not found, the last key of the dictionary will be used as the target.
            seed (`int`, *optional*, defaults to `None`):
                Seed for sampler.
        """
        self.dataset = dataset
        self.temperature = temperature
        self.distribution = distribution
        self.seed = seed
        self.exponent = 1 / temperature
        self.index_or_key = None
        if target_format is None:
            _getitem_class = type(dataset[0])
            if _getitem_class is dict:
                keys = list(dataset[0].keys())
                index = next((key for key in keys if key in {"label", "target"}), None)
                if index is None: index = keys[-1]
            elif _getitem_class in {tuple, list}:
                index = -1
            else:
                raise AttributeError(
                    f"Class type '{_getitem_class}' is unsupported for __getitem__ Dataset function and "
                    f"TemperatureSampler sampler."
                )
            
            target_format = lambda batch: batch[index]
            
        self.target_format = target_format

    def __call__(self):
        distribution_dict = self._get_distributions()
        weights = [distribution_dict[self.target_format(sample)] for sample in self.dataset]

        generator = None
        if self.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self.seed)
        
        return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)

    def _get_distributions(self):
        total = len(self.dataset)
        distribution_dict = self.distribution
        
        if distribution_dict is None:
            targets = np.array([sample[self.index_or_key] for sample in self.dataset])
            unique, counts = np.unique(targets, return_counts=True)
            classes = dict(zip(unique, counts))
        elif isinstance(distribution_dict, str):
            ext = distribution_dict.split("/")[-1].split(".")[-1]
            # Check for Pandas format
            if ext in PANDAS_READER_MAP:
                df = PANDAS_READER_MAP[ext](distribution_dict)
                cols = df.columns.tolist()
                # negative values to fix if "Index" is an extra column
                class_col = cols[-2]
                count_col = cols[-1]

                classes = dict(zip(df[class_col].tolist(), df[count_col].tolist()))
            elif ext in {"json", "yaml", "yml"}:
                classes = yaml.safe_load(open(distribution_dict))

        keys = np.array(list(classes.keys()))
        values = np.array(list(classes.values()), dtype=float)
        values = (values / total) ** self.exponent
        distribution_dict = dict(zip(keys, values))

        return distribution_dict
