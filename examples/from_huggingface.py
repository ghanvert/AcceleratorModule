# Copyright 2022 ghanvert. All rights reserved.
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

from dummy_dataset import DummyTranslationDataset
from transformers import NllbTokenizer

from accmt import DataCollatorForSeq2Seq, HyperParameters, Trainer


hf_model = "facebook/nllb-200-distilled-600M"

tokenizer = NllbTokenizer.from_pretrained(hf_model)
train_dataset = DummyTranslationDataset(tokenizer)

trainer = Trainer(
    hps_config=HyperParameters(),
    model_path="dummy_model",
    collate_fn=DataCollatorForSeq2Seq(tokenizer),
    # adds efficient padding to the longest sequence in the batch.
)
trainer.fit((hf_model, "AutoModelForSeq2SeqLM"), train_dataset)
