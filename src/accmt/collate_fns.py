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
from torch.utils.data._utils.collate import default_collate
from transformers import PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase


def collate_tokenizer_inputs(batch: list, pad_token_id: int, label_pad_token_id: int, padding_side: str):
    include_attention_mask = "attention_mask" in batch[0]
    include_labels = "labels" in batch[0]

    inputs = []
    labels = []
    for feature in batch:
        inputs.append(len(feature["input_ids"]))
        if include_labels:
            labels.append(len(feature["labels"]))

    max_input_length = max(inputs)
    if include_labels:
        max_label_length = max(labels)

    inputs = []
    attention_masks = []
    labels = []
    for feature in batch:
        inputs_remainder = [pad_token_id] * (max_input_length - len(feature["input_ids"]))
        if include_attention_mask:
            attention_masks_remainder = [0] * (max_input_length - len(feature["input_ids"]))
        if include_labels:
            labels_remainder = [label_pad_token_id] * (max_label_length - len(feature["labels"]))

        if include_labels and isinstance(feature["labels"], list):
            feature = {
                "input_ids": feature["input_ids"] + inputs_remainder,
                "attention_mask": (
                    feature["attention_mask"] + attention_masks_remainder if include_attention_mask else None
                ),
                "labels": (feature["labels"] + labels_remainder),
            }
        elif padding_side == "right":
            feature = {
                "input_ids": np.concatenate([feature["input_ids"], inputs_remainder]).astype(np.int64),
                "attention_mask": (
                    np.concatenate([feature["attention_mask"], attention_masks_remainder]).astype(np.int64)
                    if include_attention_mask
                    else None
                ),
                "labels": (
                    np.concatenate([feature["labels"], labels_remainder]).astype(np.int64) if include_labels else None
                ),
            }
        else:
            feature = {
                "input_ids": np.concatenate([inputs_remainder, feature["input_ids"]]).astype(np.int64),
                "attention_mask": (
                    np.concatenate([attention_masks_remainder, feature["attention_mask"]]).astype(np.int64)
                    if include_attention_mask
                    else None
                ),
                "labels": (
                    np.concatenate([labels_remainder, feature["labels"]]).astype(np.int64) if include_labels else None
                ),
            }

        inputs.append(feature["input_ids"])
        if include_attention_mask:
            attention_masks.append(feature["attention_mask"])

        if include_labels:
            labels.append(feature["labels"])

    output_dict = {"input_ids": torch.from_numpy(np.stack(inputs))}

    if include_attention_mask:
        output_dict["attention_mask"] = torch.from_numpy(np.stack(attention_masks))

    if include_labels:
        output_dict["labels"] = torch.from_numpy(np.stack(labels))

    return output_dict


# function derived from 'transformers' library: https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L52
def pad_without_fast_tokenizer_warning(tokenizer, *pad_args, **pad_kwargs):
    """
    Pads without triggering the warning about how using the pad function is sub-optimal when using a fast tokenizer.
    """

    # To avoid errors when using Feature extractors
    if not hasattr(tokenizer, "deprecation_warnings"):
        return tokenizer.pad(*pad_args, **pad_kwargs)

    # Save the state of the warning, then disable it
    warning_state = tokenizer.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    try:
        padded = tokenizer.pad(*pad_args, **pad_kwargs)
    finally:
        # Restore the state of the warning.
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = warning_state

    return padded


def stack_tensor_dict(tensor_dicts: list[dict[torch.Tensor]]):
    keys = tensor_dicts[0].keys()
    return {key: torch.stack([d[key] for d in tensor_dicts]) for key in keys}


def stack_iterables(iterables: list[list | tuple]):
    return torch.stack([torch.tensor(iterable) for iterable in iterables])


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
    model: Optional[PreTrainedModel] = None
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


class DataCollatorForLongestSequence:
    """
    Automatically adds efficient padding for inputs, while preserving static labels.

    If output of `__getitem__` Dataset logic looks like:
        `return x, y` (x being a dictionary containing keys `input_ids` and `attention_mask`)
    then the output of the collator function will be `(x, y)`, `x` being the padded inputs with
    the same keys and `y` the stacked labels.

    If output of `__getitem__` Dataset logic looks like:
        `return x` (x being a dictionary containing keys `input_ids` and `attention_mask`)
    then the output of the collator function will be `x`, being the padded inputs with the same keys.

    NOTE: This collator should be used when labels on your dataset logic are not sequences. If that's the case,
    see `DataCollatorForSeq2Seq`.

    Args:
        tokenizer (`Any`):
            Tokenizer using HuggingFace standard.
    """

    def __init__(self, tokenizer: Any, torch_stack: bool = True):
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        self.padding_side = self.tokenizer.padding_side
        self.torch_stack = torch_stack
        self.device = None

    def __call__(self, batch: list):
        inputs = []
        for feature in batch:
            # if feature is a tuple, then it would be of type (inputs, targets)
            if isinstance(feature, tuple):
                feature = feature[0]  # just take first element
            inputs.append(len(feature["input_ids"]))

        max_input_length = max(inputs)

        inputs = []
        attention_masks = []
        labels = []
        for feature in batch:
            if isinstance(feature, tuple):
                labels.append(feature[1])
                feature = feature[0]
            inputs_remainder = [self.pad_token_id] * (max_input_length - len(feature["input_ids"]))
            attention_masks_remainder = [0] * (max_input_length - len(feature["input_ids"]))

            if self.padding_side == "right":
                feature = {
                    "input_ids": np.concatenate([feature["input_ids"], inputs_remainder]).astype(np.int64),
                    "attention_mask": np.concatenate([feature["attention_mask"], attention_masks_remainder]).astype(
                        np.int64
                    ),
                }
            else:
                feature = {
                    "input_ids": np.concatenate([inputs_remainder, feature["input_ids"]]).astype(np.int64),
                    "attention_mask": np.concatenate([attention_masks_remainder, feature["attention_mask"]]).astype(
                        np.int64
                    ),
                }

            inputs.append(feature["input_ids"])
            attention_masks.append(feature["attention_mask"])

        output = {
            "input_ids": torch.from_numpy(np.stack(inputs)),
            "attention_mask": torch.from_numpy(np.stack(attention_masks)),
        }

        if len(labels) > 0:
            if isinstance(labels[0], dict):
                keys = labels[0].keys()
                if self.torch_stack:
                    out_labels = {k: torch.stack([label[k] for label in labels]) for k in keys}
                else:
                    out_labels = {k: [label[k] for label in labels] for k in keys}
            elif isinstance(labels[0], torch.Tensor):
                out_labels = labels
                if self.torch_stack:
                    out_labels = torch.stack(out_labels)
            elif isinstance(labels[0], (list, tuple, np.ndarray)):
                out_labels = [torch.tensor(label, device=self.device) for label in labels]
                if self.torch_stack:
                    out_labels = torch.stack(out_labels)
            else:
                out_labels = None

            return output, out_labels

        return output


class DataCollatorForLanguageModeling:
    """
    Collator function to implement automatic language modeling, such as
    Masked Language Modeling.

    Args:
        tokenizer (`Any`):
            Tokenizer using HuggingFace standard.
        mlm (`bool`, *optional*, defaults to `True`):
            Implements Masked Language Modeling.
        mlm_probability (`float`, *optional*, defaults to `0.15`):
            How much masking is implemented in Masked Language Modeling.
        ignore_index (`int`, *optional*, defaults to `-100`):
            Label pad token id. Labels with this value will be ignored in the training process.
        masked_to_mask (`float`, *optional*, defaults to `0.8`):
            Probability to replace masked input tokens with mask token. The half remaining percent will
            replace masked input tokens with random word, and the other half will keep the masked input tokens
            unchanged. If `apply_random_words` is set to `False`, then the entire remaining percent will be unchanged.
        apply_random_words (`bool`, *optional*, defaults to `True`):
            Whether to apply random words during Masked Language Modeling.
        force_one_output (`bool`, *optional*, defaults to `False`):
            Whether to force output one output. If Dataset object `__getitem__` function returns a tuple, only the first
            element will be considered and extra targets will be dropped.
    """

    def __init__(
        self,
        tokenizer: Any,
        mlm: bool = True,
        mlm_probability: float = 0.15,
        ignore_index: int = -100,
        masked_to_mask: float = 0.8,
        apply_random_words: bool = True,
        force_one_output: bool = False,
    ) -> Union[dict, tuple[dict, torch.Tensor]]:
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.mlm_probability = mlm_probability
        self.ignore_index = ignore_index
        self.masked_to_mask = masked_to_mask
        self.apply_random_words = apply_random_words
        self.force_one_output = force_one_output

    def __call__(self, batch: list) -> dict:
        has_extra_targets = isinstance(batch[0], (tuple, list))
        if not has_extra_targets:
            tokenizer_dict = pad_without_fast_tokenizer_warning(self.tokenizer, batch, return_tensors="pt")
        else:
            tokenizer_dict_batch, extra_targets = [], []
            for elems in batch:
                tokenizer_dict_batch.append(elems[0])
                if not self.force_one_output:
                    extra_targets.append(elems[1:])

            tokenizer_dict = pad_without_fast_tokenizer_warning(
                self.tokenizer, tokenizer_dict_batch, return_tensors="pt"
            )

        special_tokens_mask = tokenizer_dict.pop("special_tokens_mask", None)
        if self.mlm:
            tokenizer_dict["input_ids"], tokenizer_dict["labels"] = self.torch_mask_tokens(
                tokenizer_dict["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = tokenizer_dict["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = self.ignore_index
            tokenizer_dict["labels"] = labels

        if has_extra_targets and not self.force_one_output:
            num_elems = len(extra_targets[0])
            extra_targets_return = [[] for _ in range(num_elems)]

            for target in extra_targets:
                for idx in range(num_elems):
                    tgt = target[idx]
                    extra_targets_return[idx].append(tgt)

            stack_funcs = []
            for idx, extra_target_return in enumerate(extra_targets_return):
                first_elem = extra_target_return[0]
                if isinstance(first_elem, torch.Tensor):
                    stack_funcs.append(torch.stack)
                elif isinstance(first_elem, dict):
                    stack_funcs.append(stack_tensor_dict)
                elif isinstance(first_elem, (tuple, list)):
                    stack_funcs.append(stack_iterables)
                else:
                    stack_funcs.append(torch.tensor)

            extra_targets_return = [
                stack_funcs[idx](extra_target_return) for idx, extra_target_return in enumerate(extra_targets_return)
            ]

        if has_extra_targets and not self.force_one_output:
            return tokenizer_dict, *extra_targets_return

        return tokenizer_dict

    def torch_mask_tokens(self, inputs: torch.Tensor, special_tokens_mask):
        labels = inputs.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = self.ignore_index

        indices_replaced = torch.bernoulli(torch.full(labels.shape, self.masked_to_mask)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        if self.apply_random_words:
            indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
            inputs[indices_random] = random_words[indices_random]

        return inputs, labels
