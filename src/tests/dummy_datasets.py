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

from datasets import load_dataset
from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    def __init__(self, tokenizer, max_length: int = 128, is_val: bool = False):
        self.dataset = load_dataset("opus_books", "en-es")["train"]
        _range = range(1000) if is_val else range(1000, 2000)
        self.dataset = self.dataset.select(_range)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._tokenizer_kwargs = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "max_length": max_length,
        }

    def __len__(self):
        return len(self.dataset) * 2

    def __getitem__(self, idx):
        sample = self.dataset[idx % len(self.dataset)]

        src_lang, tgt_lang = ("es", "en") if idx >= len(self.dataset) else ("en", "es")

        src_text = sample[src_lang]
        tgt_text = sample[tgt_lang]

        src_encoding = self.tokenizer(src_text, **self._tokenizer_kwargs)
        tgt_encoding = self.tokenizer(tgt_text, **self._tokenizer_kwargs)

        output_dict = {
            "input_ids": src_encoding["input_ids"],
            "attention_mask": src_encoding["attention_mask"],
            "labels": tgt_encoding["input_ids"],
        }

        return src_lang, tgt_lang, output_dict
