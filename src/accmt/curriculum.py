# Copyright 2025 ghanvert. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from typing import Optional, Union

from torch.utils.data import DataLoader, Dataset
from typing_extensions import override


class _CurriculumLearning(ABC):
    """
    Base class for curriculum learning.
    """

    def __init__(self):
        self.steps_per_data: list[int] = []
        self.data: list[Union[Dataset, DataLoader]] = []
        self.dataloader_kwargs: list[dict] = []
        self._datasets_converted: bool = False

    def convert_datasets_to_dataloaders(self, **kwargs):
        """
        Convert `Dataset` instances to `DataLoader`.

        Args:
            **kwargs (`Any`):
                Keyword arguments for `DataLoader`.
        """
        for i, data in enumerate(self.data):
            if isinstance(data, Dataset):
                # update global dataloader kwargs without modifying the original dict
                dl_kwargs = {**kwargs}
                dl_kwargs.update(self.dataloader_kwargs[i])
                self.data[i] = DataLoader(data, **dl_kwargs)
            else:
                raise ValueError(f"Data at index {i} is not a `Dataset` instance.")

        self._datasets_converted = True

    @abstractmethod
    def add(self, *args, **kwargs):
        pass

    @abstractmethod
    def convert_to_max_step_per_dataloader(
        self, training_max_steps: Optional[int]
    ) -> list[tuple[int, DataLoader, dict]]:
        """
        Get a list of tuples of `(max_step, dataloader)` for each dataloader.
        NOTE: If you want to later know the last index step per dataloader, you need to substract 1 from the returned value.

        Args:
            training_max_steps (`int`):
                Maximum steps for training.

        Returns:
            `list`:
                List of tuples of `(max_step, dataloader, dataloader_kwargs)` for each dataloader.
        """
        if len(self.data) == 0:
            raise RuntimeError("No data added to curriculum learning.")

        if not self._datasets_converted:
            raise RuntimeError("Must call `convert_datasets_to_dataloaders` before calling this method.")

        if training_max_steps is None:
            raise ValueError("`max_steps` in hyperparameters must be specified when using curriculum learning.")

    def __len__(self) -> int:
        return len(self.data)


class StepsCurriculum(_CurriculumLearning):
    """
    Curriculum learning with fixed steps per dataset.
    """

    def __init__(self):
        super().__init__()

    @override
    def add(self, dataset: Dataset, steps: int, dataloader_kwargs: Optional[dict] = None):
        """
        Add a dataset with fixed steps.

        Args:
            dataset (`Dataset`):
                Dataset to add.
            steps (`int`):
                Number of steps for the dataset.
            dataloader_kwargs (`dict`, *optional*, defaults to `None`):
                Keyword arguments for the dataloader.
        """
        if steps == 0 or steps < -1:
            raise ValueError("Steps must be greater than 0 or -1 (dynamic sizing for the last dataset).")

        self.steps_per_data.append(steps)
        self.data.append(dataset)
        self.dataloader_kwargs.append(dataloader_kwargs or {})

    @override
    def convert_to_max_step_per_dataloader(
        self, training_max_steps: Optional[int]
    ) -> list[tuple[int, DataLoader, dict]]:
        super().convert_to_max_step_per_dataloader(training_max_steps)
        _list = []
        cumsum = 0
        for i, (data, steps, dataloader_kwargs) in enumerate(
            zip(self.data, self.steps_per_data, self.dataloader_kwargs)
        ):
            if i != len(self.data) - 1 and steps == -1:
                raise ValueError("-1 is not allowed for a non-last dataset in curriculum learning.")

            cumsum += steps if steps != -1 else training_max_steps - cumsum
            if cumsum > training_max_steps:
                raise ValueError(
                    "You are exceeding the maximum number of steps for training when declaring curriculum learning. "
                    f"`max_steps` is {training_max_steps}. Consider using -1 for the last dataset to enable dynamic "
                    "sizing."
                )

            _list.append((cumsum, data, dataloader_kwargs))

        return _list


class RatioCurriculum(_CurriculumLearning):
    """
    Curriculum learning with fixed ratio (based in `max_steps`) per dataset.
    """

    def __init__(self):
        super().__init__()
        self.ratios: list[float] = []

    @override
    def add(self, dataset: Dataset, ratio: float, dataloader_kwargs: Optional[dict] = None):
        """
        Add a dataset with fixed ratio.

        Args:
            dataset (`Dataset`):
                Dataset to add.
            ratio (`float`):
                Ratio for the dataset.
            dataloader_kwargs (`dict`, *optional*, defaults to `None`):
                Keyword arguments for the dataloader.
        """
        if ratio != -1 and (ratio <= 0 or ratio > 1):
            raise ValueError("Ratio must be between 0 and 1, or -1 (dynamic sizing for the last dataset).")
        elif ratio == -1:
            ratio = 1 - sum(self.ratios)

        if sum(self.ratios) + ratio > 1:
            raise ValueError("Sum of ratios must be less than or equal to 1.")

        self.ratios.append(ratio)
        self.data.append(dataset)
        self.dataloader_kwargs.append(dataloader_kwargs or {})

    @override
    def convert_to_max_step_per_dataloader(
        self, training_max_steps: Optional[int]
    ) -> list[tuple[int, DataLoader, dict]]:
        super().convert_to_max_step_per_dataloader(training_max_steps)
        _list = []
        cumsum = 0
        for data, ratio, dataloader_kwargs in zip(self.data, self.ratios, self.dataloader_kwargs):
            cumsum += ratio * training_max_steps if ratio != -1 else training_max_steps - cumsum
            if cumsum > training_max_steps:
                raise ValueError(
                    "You are exceeding the maximum number of steps for training when declaring curriculum learning. "
                    f"`max_steps` is {training_max_steps}. Consider using -1 for the last dataset to enable dynamic "
                    "sizing."
                )

            _list.append((cumsum, data, dataloader_kwargs))

        return _list


class RangeCurriculum(_CurriculumLearning):
    """
    Curriculum learning with fixed range per dataset.
    """

    def __init__(self):
        super().__init__()
        self.ranges: list[range] = []

    @override
    def add(self, dataset: Dataset, range: range, dataloader_kwargs: Optional[dict] = None):
        """
        Add a dataset with fixed range.

        Args:
            dataset (`Dataset`):
                Dataset to add.
            range (`range`):
                Range for the dataset.
            dataloader_kwargs (`dict`, *optional*, defaults to `None`):
                Keyword arguments for the dataloader.
        """
        # -1 is not allowed for ranges, thus last dataset must match training max steps
        if range.start < 0 or range.stop < 0:
            raise ValueError("Range must be non-negative.")

        if range.start >= range.stop:
            raise ValueError("Start of range must be less than the end of range.")

        if len(self.ranges) > 0 and range.start != self.ranges[-1].stop:
            raise ValueError("Start of range must be equal to the end of the previous range.")

        self.ranges.append(range)
        self.data.append(dataset)
        self.dataloader_kwargs.append(dataloader_kwargs or {})

    @override
    def convert_to_max_step_per_dataloader(
        self, training_max_steps: Optional[int]
    ) -> list[tuple[int, DataLoader, dict]]:
        super().convert_to_max_step_per_dataloader(training_max_steps)
        _list = []
        cumsum = 0
        for i, (data, _range, dataloader_kwargs) in enumerate(zip(self.data, self.ranges, self.dataloader_kwargs)):
            if i == len(self.data) - 1 and _range.stop != training_max_steps:
                raise ValueError("The last dataset must match the training max steps.")

            cumsum += _range.stop - _range.start
            if cumsum > training_max_steps:
                raise ValueError(
                    "You are exceeding the maximum number of steps for training when declaring curriculum learning. "
                    f"`max_steps` is {training_max_steps}. Consider using -1 for the last dataset to enable dynamic "
                    "sizing."
                )

            _list.append((cumsum, data, dataloader_kwargs))

        return _list
