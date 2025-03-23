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

import torch
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    def __init__(self):
        self.dataset = [([3, 2], [1, 0, 0]), ([2, 1], [0, 0, 1]), ([2, 0], [0, 1, 0]), ([4, 3], [1, 0, 0])] * 4

    def __getitem__(self, idx):
        input = torch.tensor(self.dataset[idx][0], dtype=torch.float32)
        target = torch.tensor(self.dataset[idx][1], dtype=torch.float32)

        return input, target

    def __len__(self):
        return len(self.dataset)


class DummyTranslationDataset(Dataset):
    def __init__(self, tokenizer, max_length=400):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self._tokenizer_args = {
            "return_tensors": "pt",
            "truncation": True,
            "max_length": self.max_length,
            # "padding": "max_length"  # disable this because we will use a DataCollator.
        }
        self.src_lang = "eng_Latn"
        self.tgt_lang = "spa_Latn"
        self.tokenizer.src_lang = self.src_lang
        self.tokenizer.tgt_lang = self.tgt_lang
        self.translate_from, self.translate_to = ("eng_Latn", "spa_Latn")
        self.dataset = [
            ("Hello World!", "Hola Mundo!"),
            ("How are you?", "Cómo estás?"),
            ("Are you feeling okay?", "Te sientes bien?"),
            ("That looks really great on you.", "Eso luce muy bien en ti."),
        ]

    def __getitem__(self, idx):
        source_str = self.dataset[idx][0]
        translation_str = self.dataset[idx][1]
        model_inputs = self.tokenizer(source_str, text_target=translation_str, **self._tokenizer_args)

        for key in model_inputs:
            model_inputs[key] = model_inputs[key].squeeze(0)

        return model_inputs

    def __len__(self):
        return len(self.dataset)
