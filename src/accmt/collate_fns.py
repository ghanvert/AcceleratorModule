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
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import torch
import transformers
from torch.utils.data._utils.collate import default_collate
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase


class BaseDataCollator(ABC):
    @abstractmethod
    def collate_tokenizer_inputs(self, batch: list[Union[BatchEncoding, dict]]) -> Union[tuple, Any]:
        pass

    def __call__(self, batch: list) -> Union[tuple, Any]:
        first_elem = batch[0]
        if isinstance(first_elem, (dict, BatchEncoding)):
            if "input_ids" in first_elem:
                return self.collate_tokenizer_inputs(batch)
            else:
                collated_dict = {}
                for key in first_elem.keys():
                    values = [elem[key] for elem in batch]
                    collated_dict[key] = self.__call__(values)
                return collated_dict
        elif isinstance(first_elem, (list, tuple)):
            return type(first_elem)(self.__call__(list(elem)) for elem in zip(*batch))

        return default_collate(batch)


@dataclass
class DataCollatorForSeq2Seq(BaseDataCollator):
    """
    `DataCollatorForSeq2Seq` from `transformers` library with a recursive approach to handle diverse structures.
    For documentation, refer to `DataCollatorForSeq2Seq` class
    (https://huggingface.co/docs/transformers/main_classes/data_collator#transformers.DataCollatorForSeq2Seq).
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def collate_tokenizer_inputs(self, features: list[Union[BatchEncoding, dict]]):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        # reconvert list[None] to None if necessary
        # this might occur when we pass {..., "labels": None}
        if labels is not None and all(label is None for label in labels):
            labels = None
        non_labels_features = [{k: v for k, v in feature.items() if k != label_name} for feature in features]

        # run through tokenizer without labels to ensure no side effects
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            non_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # we have to pad the labels manually as we cannot rely on `tokenizer.pad` and we need them to be of the same length to return tensors
        no_padding = self.padding is False or self.padding == PaddingStrategy.DO_NOT_PAD
        if labels is not None:
            if no_padding:
                if isinstance(features[0][label_name], list):
                    batch["labels"] = list(labels)
                else:
                    batch["labels"] = [np.concatenate([label, []]) for label in labels]
            else:
                max_padding = self.padding == PaddingStrategy.MAX_LENGTH and self.max_length is not None
                max_label_length = max(len(l) for l in labels) if not max_padding else self.max_length
                if self.pad_to_multiple_of is not None:
                    max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                    )

                padding_side = self.tokenizer.padding_side
                if isinstance(features[0][label_name], list):
                    batch["labels"] = [
                        label + [self.label_pad_token_id] * (max_label_length - len(label))
                        if padding_side == "right"
                        else [self.label_pad_token_id] * (max_label_length - len(label)) + label
                        for label in labels
                    ]
                else:
                    batch["labels"] = [
                        np.concatenate(
                            [
                                label,
                                np.array([self.label_pad_token_id] * (max_label_length - len(label)), dtype=np.int64),
                            ]
                        )
                        if padding_side == "right"
                        else np.concatenate(
                            [
                                np.array([self.label_pad_token_id] * (max_label_length - len(label)), dtype=np.int64),
                                label,
                            ]
                        )
                        for label in labels
                    ]

        # reintroduce side effects via tokenizer that return respective datatypes for the `return_tensors` argument
        if batch.get("labels", None) is not None:
            batch["labels"] = torch.from_numpy(np.array(batch["labels"], dtype=np.int64))
        else:
            batch["labels"] = None

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=batch["labels"])
            batch["decoder_input_ids"] = decoder_input_ids

        return batch


@dataclass
class DataCollatorWithPadding(BaseDataCollator, transformers.DataCollatorWithPadding):
    """
    `DataCollatorWithPadding` from `transformers` library with a recursive approach to handle diverse structures.
    For documentation, refer to `DataCollatorWithPadding` class
    (https://huggingface.co/docs/transformers/main_classes/data_collator#transformers.DataCollatorWithPadding).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def collate_tokenizer_inputs(self, features: list[Union[BatchEncoding, dict]]):
        return transformers.DataCollatorWithPadding.__call__(self, features)


@dataclass
class DataCollatorForTokenClassification(BaseDataCollator, transformers.DataCollatorForTokenClassification):
    """
    `DataCollatorForTokenClassification` from `transformers` library with a recursive approach to handle diverse structures.
    For documentation, refer to `DataCollatorForTokenClassification` class
    (https://huggingface.co/docs/transformers/main_classes/data_collator#transformers.DataCollatorForTokenClassification).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def collate_tokenizer_inputs(self, features: list[Union[BatchEncoding, dict]]):
        return transformers.DataCollatorForTokenClassification.__call__(self, features)


@dataclass
class DataCollatorForLanguageModeling(BaseDataCollator, transformers.DataCollatorForLanguageModeling):
    """
    `DataCollatorForLanguageModeling` from `transformers` library with a recursive approach to handle diverse structures.
    For documentation, refer to `DataCollatorForLanguageModeling` class
    (https://huggingface.co/docs/transformers/main_classes/data_collator#transformers.DataCollatorForLanguageModeling).
    """

    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def collate_tokenizer_inputs(self, features: list[Union[BatchEncoding, dict]]):
        return transformers.DataCollatorForLanguageModeling.__call__(self, features)


@dataclass
class DataCollatorForWholeWordMask(BaseDataCollator, transformers.DataCollatorForWholeWordMask):
    """
    `DataCollatorForWholeWordMask` from `transformers` library with a recursive approach to handle diverse structures.
    For documentation, refer to `DataCollatorForWholeWordMask` class
    (https://huggingface.co/docs/transformers/main_classes/data_collator#transformers.DataCollatorForWholeWordMask).
    """

    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def collate_tokenizer_inputs(self, features: list[Union[BatchEncoding, dict]]):
        return transformers.DataCollatorForWholeWordMask.__call__(self, features)


class DataCollatorForPermutationLanguageModeling(
    BaseDataCollator, transformers.DataCollatorForPermutationLanguageModeling
):
    """
    `DataCollatorForPermutationLanguageModeling` from `transformers` library with a recursive approach to handle diverse structures.
    For documentation, refer to `DataCollatorForPermutationLanguageModeling` class
    (https://huggingface.co/docs/transformers/main_classes/data_collator#transformers.DataCollatorForPermutationLanguageModeling).
    """

    tokenizer: PreTrainedTokenizerBase
    plm_probability: float = 1 / 6
    max_span_length: int = 5  # maximum length of a span of masked tokens
    return_tensors: str = "pt"

    def collate_tokenizer_inputs(self, features: list[Union[BatchEncoding, dict]]):
        return transformers.DataCollatorForPermutationLanguageModeling.__call__(self, features)
