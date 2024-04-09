import numpy as np
import torch

class DataCollatorForSeq2Seq:
    """
    Automatically adds efficient padding for inputs and labels.

    When called, returns a dictionary with the following keys:
        - `input_ids`
        - `attention_mask`
        - `labels`
    """
    def __init__(self, tokenizer, label_pad_token_id=-100):
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        self.label_pad_token_id = label_pad_token_id
        self.padding_side = self.tokenizer.padding_side

    def __call__(self, batch: list) -> dict:
        inputs = []
        attention_masks = []
        labels = []
        for feature in batch:
            inputs.append(feature["input_ids"])
            attention_masks.append(feature["attention_mask"])
            labels.append(feature["labels"])

        max_label_length = max(len(l) for l in labels)
        max_input_length = max(len(l) for l in inputs)

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
