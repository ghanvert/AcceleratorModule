import yaml
import torch
import numpy as np
import os
import warnings
from abc import ABC, abstractmethod
from pympler import asizeof
from torch.utils.data import Dataset, Sampler
from typing_extensions import Any, Optional, Union, Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Sequence, Iterator
from numba import njit, prange
from numba.core import types
from numba.typed import Dict
from .utils import PANDAS_READER_MAP, divide_list

class DistributedWeightedRandomSampler(Sampler[int]):
    r"""Samples elements from ``[0,..,len(weights)-1]`` with given probabilities (weights).

    Args:
        weights (sequence): a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
        generator (Generator): Generator used in sampling.

    Example:
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> list(WeightedRandomSampler([0.1, 0.9, 0.4, 0.7, 3.0, 0.6], 5, replacement=True))
        [4, 4, 1, 4, 5]
        >>> list(WeightedRandomSampler([0.9, 0.4, 0.05, 0.2, 0.3, 0.1], 5, replacement=False))
        [0, 1, 4, 3, 2]

    This implementation derives from https://github.com/huggingface/accelerate/issues/2865#issuecomment-2175681615 
    GitHub user: FrsECM. This avoids data overlap between processes in distributed training.
    """
    def __init__(self,
                 accelerator: Any,
                 weights: Sequence[float],
                 num_samples: int,
                 replacement: bool = True,
                 generator: torch.Generator = None
    ):
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or num_samples <= 0:
            raise ValueError(f"'num_samples' should be a positive integer value, but got num_samples={num_samples}")
        if not isinstance(replacement, bool):
            raise ValueError(f"'replacement' should be a boolean value, but got replacement={replacement}")

        # We generate a random permutation of indices.
        self.indices = torch.randperm(num_samples, generator=generator)
        # We generate weight tensor
        weights_tensor = torch.as_tensor(weights, dtype=torch.float32)[self.indices]
        if len(weights_tensor.shape) != 1:
            raise ValueError(f"'weights' should be a 1D sequence but given weights have shape {tuple(weights_tensor.shape)}")
        self.mask = torch.ones_like(weights_tensor).bool()

        num_processes = accelerator.num_processes
        if num_processes > 1:
            assert generator is not None, "A generator should be set when num_processes > 1"
            # We reset the mask to zero for all processes
            self.mask = torch.zeros_like(weights_tensor)
            # We want the mask to select only indices for the current process
            # => We cut our indices in num_processes parts and we set the mask to 1 where the rank is matching
            rank_indices = [i for i in range(len(self.mask)) if i % num_processes == accelerator.process_index]
            self.mask[rank_indices] = 1 
            self.mask = self.mask.bool()

        # Set parameters...
        self.weights = weights_tensor
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        # We sample "num_samples" indices from the weights tensor "masked" on current process weights
        rand_tensor = torch.multinomial(self.weights[self.mask], self.num_samples, self.replacement, generator=self.generator)
        # We get corresponding indices
        rank_indices = self.indices[self.mask]
        rand_indices = rank_indices[rand_tensor]
        rand_indices: torch.Tensor
        # We sample only from theses indices.
        yield from iter(rand_indices.tolist())

    def __len__(self):
        return self.num_samples

class BaseSampler(ABC):
    @abstractmethod
    def __call__(self, accelerator):
        pass

@njit(parallel=True)
def process_strings(strings, values, result):
    for i in prange(len(strings)):
        result[i] = values[strings[i]]

def numba_process(dataset: list, distribution_dict: dict):
    int_dtype = (np.int16, types.int16) if len(distribution_dict) <= 32767 else (np.int32, types.int32)
    d = Dict.empty(key_type=int_dtype[1], value_type=types.float32)
    if isinstance(dataset[0], str):
        unique_classes = distribution_dict.keys()
        str_to_int = {cls:i for i, cls in enumerate(unique_classes)}
        samples = np.fromiter(map(str_to_int.__getitem__, dataset), dtype=int_dtype[0], count=len(dataset))
        for k, v in distribution_dict.items(): d[str_to_int[k]] = v
    elif isinstance(dataset[0], int):
        samples = np.array(dataset, dtype=int_dtype)
        for k, v in distribution_dict.items(): d[k] = v
    result = np.empty(len(dataset), dtype=np.float32)
    values = np.array([d[i] for i in range(len(d))], dtype=np.float32)
    process_strings(samples, values, result)
    return result

class TemperatureSampler(BaseSampler):
    def __init__(self,
                 dataset: Union[Dataset, Iterable],
                 temperature: int,
                 distribution: Optional[Union[dict, str]] = None,
                 target_format: Optional[Callable] = None,
                 seed: Optional[int] = None,
                 use_threads: bool = True
    ):
        """
        Temperature sampler based on the up-sampling technique described in 
        "Massively Multilingual Neural Machine Translation in the Wild: Findings 
        and Challenges" (https://arxiv.org/pdf/1907.05019).

        Useful for imbalanced datasets to up-sample low-resource classes.

        NOTE: When iterating over the dataset, this looks for the biggest attribute in memory (usually the raw dataset) 
        in the `dataset` object. This is done to not use `__getitem__` from Dataset, which may have some extra 
        processing that will slow the process of constructing the sampler. If somehow the biggest attribute in memory 
        is not found in the dataset object, you may need to create an `Iterable` object (like a `list`) which contains 
        the raw formatted dataset, and also specify the format with `target_format` on how to get access to the information.

        Args:
            dataset (`Dataset` or `Iterable`):
                PyTorch `Dataset` object, or an `Iterable` (like a `list`) containing the raw formatted dataset.
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
                Target format when iterating over the dataset. This indicates the format 
                on how to get access to every sample's label in order to calculate the corresponding 
                sample weight (probability).

                Examples:
                    target_format = lambda sample: sample[1] # indicates that the second element of the `__getitem__` return 
                    value is the class target.

                    target_format = lambda sample: sample[0]['language'] # indicates that the first element of the `__getitem__` 
                    return value is a dictionary containing the key 'language', which is the class target.
                
                If not specified, the target format will be inferred based on the first element of the dataset. It will get 
                the return type of `__getitem__`, and in the case it's a `tuple` or `list`, the last element of it will be 
                considered as the target. If the return type is a `dict`, it will look for keys 'label' or 'target'. If these 
                keys are not found, the last key of the dictionary will be used as the target.
            seed (`int`, *optional*, defaults to `None`):
                Seed for sampler.
            use_threads (`bool`, *optional*, defaults to `True`):
                Enable the use of threads for faster processing when calculating weights and iterating over the dataset.
        """
        if isinstance(dataset, Dataset):
            attributes = dir(dataset)
            larger_value = 0
            larger_attr = "default"
            for attr in attributes:
                if "__" in attr: continue
                size = asizeof.asizeof(getattr(dataset, attr))
                if size > larger_value:
                    larger_value = size
                    larger_attr = attr
            
            dataset = getattr(dataset, larger_attr)
        
        self.dataset = dataset
        self.temperature = temperature
        self.distribution = distribution
        self.seed = seed
        self.exponent = 1 / temperature
        self.use_threads = use_threads
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
            
            target_format = lambda sample: sample[index]
            
        self.target_format = target_format

    def __call__(self, accelerator):
        effective_num_threads = os.cpu_count() // accelerator.num_processes
        effective_num_threads = effective_num_threads if effective_num_threads >= 4 else 4
        num_threads = effective_num_threads if self.use_threads and effective_num_threads > 0 else 1
        distribution_dict = self._get_distributions(num_threads)

        weights = numba_process(self.target_format(self.dataset), distribution_dict)

        generator = None
        if self.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self.seed)
        
        return DistributedWeightedRandomSampler(
            accelerator, weights=weights, num_samples=len(weights), replacement=True, generator=generator
        )

    def _get_distributions(self, num_threads):
        total = len(self.dataset)
        distribution_dict = self.distribution
        
        if distribution_dict is None:
            # TODO: Dividing dataset and parallelizing process is slow. Maybe we can use numba.
            warnings.warn(
                "Calculating class distribution for big datasets might be really slow. We will fully "
                "support this in a future version. For now, please provide a distribution dictionary or "
                "path to read the distribution from. See the documentation for supported formats."
            )
            # Also this might be incorrect, since 'self.dataset' could be a dictionary, so we could 
            # do 'self.target_format(self.dataset)' directly without dividing the dataset for multi-threading.
            divided_dataset = divide_list(self.dataset, num_threads)
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                results = list(executor.map(self.target_format, divided_dataset))
            
            targets = []
            for result in results: targets.extend(result)
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
