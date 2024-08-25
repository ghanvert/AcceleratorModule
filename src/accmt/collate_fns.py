import numpy as np
import torch
from collections import defaultdict
from typing import Any, Union

class DataCollatorForSeq2Seq:
    """
    Automatically adds efficient padding for inputs and labels.

    When called, returns a dictionary with the following keys:
        - `input_ids`
        - `attention_mask`
        - `labels`

    This implementation derives from `transformers` library:
    https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L543

    Args:
        tokenizer (`Any`):
            Tokenizer using HuggingFace standard.
        label_pad_token_id (`int`, *optional*, defaults to `-100`):
            Label pad token id. Labels with this value will be ignored in the training process.
    """
    def __init__(self, tokenizer: Any, label_pad_token_id: int = -100):
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        self.label_pad_token_id = label_pad_token_id
        self.padding_side = self.tokenizer.padding_side

    def __call__(self, batch: list) -> dict:
        inputs = []
        labels = []
        for feature in batch:
            inputs.append(len(feature["input_ids"]))
            labels.append(len(feature["labels"]))

        max_label_length = max(labels)
        max_input_length = max(inputs)

        inputs = []
        attention_masks = []
        labels = []
        for feature in batch:
            inputs_remainder = [self.pad_token_id] * (max_input_length - len(feature["input_ids"]))
            attention_masks_remainder = [0] * (max_input_length - len(feature["input_ids"]))
            labels_remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
            
            if isinstance(feature["labels"], list):
                feature = {
                    "input_ids": feature["input_ids"] + inputs_remainder,
                    "attention_mask": feature["attention_mask"] + attention_masks_remainder,
                    "labels": feature["labels"] + labels_remainder
                }
            elif self.padding_side == "right":
                feature = {
                    "input_ids": np.concatenate([feature["input_ids"], inputs_remainder]).astype(np.int64),
                    "attention_mask": np.concatenate([feature["attention_mask"], attention_masks_remainder]).astype(np.int64),
                    "labels": np.concatenate([feature["labels"], labels_remainder]).astype(np.int64)
                }
            else:
                feature = {
                    "input_ids": np.concatenate([inputs_remainder, feature["input_ids"]]).astype(np.int64),
                    "attention_mask": np.concatenate([attention_masks_remainder, feature["attention_mask"]]).astype(np.int64),
                    "labels": np.concatenate([labels_remainder, feature["labels"]]).astype(np.int64)
                }

            inputs.append(feature["input_ids"])
            attention_masks.append(feature["attention_mask"])
            labels.append(feature["labels"])
    
        return {
            "input_ids": torch.from_numpy(np.stack(inputs)),
            "attention_mask": torch.from_numpy(np.stack(attention_masks)),
            "labels": torch.from_numpy(np.stack(labels))
        }

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
                feature = feature[0] # just take first element
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
                    "attention_mask": np.concatenate([feature["attention_mask"], attention_masks_remainder]).astype(np.int64)
                }
            else:
                feature = {
                    "input_ids": np.concatenate([inputs_remainder, feature["input_ids"]]).astype(np.int64),
                    "attention_mask": np.concatenate([attention_masks_remainder, feature["attention_mask"]]).astype(np.int64)
                }

            inputs.append(feature["input_ids"])
            attention_masks.append(feature["attention_mask"])

        output = {
            "input_ids": torch.from_numpy(np.stack(inputs)),
            "attention_mask": torch.from_numpy(np.stack(attention_masks))
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
        keep_original_input (`bool`, *optional*, defaults to `False`):
            Whether to add an extra key to the output dictionary called `unmasked_input_ids`, which is a tensor clone of 
            the original input.
            
            WARNING: This causes more memory consumption.
        force_one_output (`bool`, *optional*, defaults to `False`):
            Whether to force output one output. If Dataset object `__getitem__` function returns a tuple, only the first 
            element will be considered and extra targets will be dropped.
    """
    def __init__(self,
                 tokenizer: Any,
                 mlm: bool = True,
                 mlm_probability: float = 0.15,
                 ignore_index: int = -100,
                 masked_to_mask: float = 0.8,
                 apply_random_words: bool = True,
                 keep_original_input: bool = False,
                 force_one_output: bool = False
    ) -> Union[dict, tuple[dict, torch.Tensor]]:
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.mlm_probability = mlm_probability
        self.ignore_index = ignore_index
        self.masked_to_mask = masked_to_mask
        self.apply_random_words = apply_random_words
        self.keep_original_input = keep_original_input
        self.force_one_output = force_one_output

    def __call__(self, batch: list) -> dict:
        original_input_list = []
        input_list = []
        attention_mask_list = []
        label_list = []
        extra_targets = defaultdict(list)
        for feature in batch:
            if self.keep_original_input:
                original_input_list.append(_feature["input_ids"].clone())
            if isinstance(feature, tuple) and not self.force_one_output:
                _feature = feature[0]
                if isinstance(feature[1], dict):
                    for k, v in feature[1].items():
                        extra_targets[k].append(v)
            else:
                _feature = feature
                if self.force_one_output and isinstance(_feature, tuple):
                    _feature = _feature[0]
            inputs = _feature["input_ids"]
            special_tokens_mask = _feature.pop("special_tokens_mask", None)
            # specials tokens can be [CLS], [SEP], [PAD] or related
            if self.mlm:
                labels = _feature["input_ids"].clone()
                probability_matrix = torch.full(labels.shape, self.mlm_probability)
                if special_tokens_mask is None:
                    special_tokens_mask = self.tokenizer.get_special_tokens_mask(labels.tolist(), already_has_special_tokens=True)
                    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
                else:
                    special_tokens_mask = special_tokens_mask.bool()

                probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
                masked_indices = torch.bernoulli(probability_matrix).bool()
                labels[~masked_indices] = self.ignore_index # only compute loss on masked tokens

                if isinstance(self.masked_to_mask, float) and self.masked_to_mask != 0.0:
                    indices_replaced = torch.bernoulli(torch.full(labels.shape, self.masked_to_mask)).bool() & masked_indices
                    inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

                    if self.apply_random_words:
                        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
                        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
                        inputs[indices_random] = random_words[indices_random]

                _feature["labels"] = labels
            else:
                labels = _feature["input_ids"].clone()
                if self.tokenizer.pad_token_id is not None:
                    labels[labels == self.tokenizer.pad_token_id] = self.ignore_index
                _feature["labels"] = labels

            input_list.append(inputs)
            attention_mask_list.append(_feature["attention_mask"])
            label_list.append(labels)

        output = {
            "input_ids": torch.stack(input_list),
            "attention_mask": torch.stack(attention_mask_list),
            "labels": torch.stack(label_list)
        }

        if self.keep_original_input:
            output["unmasked_input_ids"] = torch.stack(original_input_list)

        if len(extra_targets) > 0:
            extra_targets = dict(extra_targets)
            for k, v in extra_targets.items():
                extra_targets[k] = torch.stack(v)

            return output, extra_targets
        else:
            return output
