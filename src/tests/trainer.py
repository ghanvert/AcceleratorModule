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

from accmt import HyperParameters, Trainer

from .dummy_callbacks import DummyCallback
from .dummy_datasets import SimpleDataset
from .dummy_metrics import Accuracy
from .dummy_modules import DummyClassificationModule


def test_classification_trainer():
    train_dataset = SimpleDataset()
    val_dataset = SimpleDataset()
    module = DummyClassificationModule()

    trainer = Trainer(
        hps_config=HyperParameters(batch_size=2, epochs=100, optimizer="AdamW", optim_kwargs={"lr": 0.01}),
        model_path="dummy_model",
        callbacks=[DummyCallback()],
        metrics=[Accuracy("accuracy")],
        resume=False,
    )

    trainer.fit(module, train_dataset, val_dataset)

    assert trainer.state.finished
