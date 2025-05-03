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

import torch.nn as nn
from typing_extensions import Any

from src.accmt import AcceleratorModule, ExtendedAcceleratorModule


class DummyTranslationModule(AcceleratorModule):
    def __init__(self, model: nn.Module, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = self.model.config.max_length
        self.pad_kwargs = {
            "value": self.tokenizer.pad_token_id,
            "padding": "max_length",
            "max_length": self.max_length,
        }

    def training_step(self, batch):
        _, _, inputs = batch
        output = self.model(**inputs)

        self.log({"another_loss": output.loss}, step=self.state.global_step)

        return output.loss

    def validation_step(self, batch):
        _, _, inputs = batch
        output = self.model(**inputs)
        loss = output.loss

        labels = inputs.pop("labels")
        labels[labels == -100] = self.tokenizer.pad_token_id
        predictions = self.model.generate(**inputs)
        references = labels

        predictions = self.pad(predictions, **self.pad_kwargs)
        references = self.pad(references, **self.pad_kwargs)

        return {
            "loss": loss,
            "bleu": (predictions, references),
        }


class DummyTranslationModuleWithDeclarations(DummyTranslationModule):
    def __init__(self, model: nn.Module, tokenizer):
        super().__init__(model, tokenizer)

    def get_optimizer(self, *args: Any, **kwargs: Any) -> Any:
        return super().get_optimizer(*args, **kwargs)


class DummyTranslationExtendedModule(ExtendedAcceleratorModule, DummyTranslationModule):
    def __init__(self, model: nn.Module, tokenizer):
        super().__init__(model, tokenizer)

    def training_step(self, batch):
        loss = super().training_step(batch)

        self.backward(loss)
        self.step()

        return loss
