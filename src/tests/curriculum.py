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
from accmt.curriculum import RangeCurriculum, RatioCurriculum, StepsCurriculum

from .dummy_callbacks import DummyCallback
from .dummy_datasets import SimpleDataset
from .dummy_metrics import Accuracy
from .dummy_modules import DummyClassificationModule


def get_simple_trainer(epochs=1, max_steps=None, batch_size=2, optimizer="AdamW", optim_kwargs={"lr": 0.01}):
    hps_config = HyperParameters(
        batch_size=batch_size, epochs=epochs, max_steps=max_steps, optimizer=optimizer, optim_kwargs=optim_kwargs
    )
    return Trainer(
        hps_config=hps_config,
        model_path="dummy_model",
        callbacks=[DummyCallback()],
        metrics=[Accuracy("accuracy")],
        grad_accumulation_steps=2,
        resume=False,
    )


def test_default_curriculum():
    train_dataset = [
        (10, SimpleDataset(), {"shuffle": False}),
        (20, SimpleDataset(), {"shuffle": False}),
        (30, SimpleDataset(), {"shuffle": False}),
        (40, SimpleDataset()),
        (-1, SimpleDataset(), {"shuffle": False}),
    ]
    val_dataset = SimpleDataset()
    module = DummyClassificationModule()

    trainer = get_simple_trainer(max_steps=100)
    trainer.fit(module, train_dataset, val_dataset)

    assert trainer.state.finished and trainer.state.train_dataloader_idx == len(train_dataset)
    # second assertion is correct because at the end of training the trainer does a "+ 1" to the index


def test_steps_curriculum():
    curriculum = StepsCurriculum()
    curriculum.add(SimpleDataset(), 10)
    curriculum.add(SimpleDataset(), 10, {"shuffle": False})
    curriculum.add(SimpleDataset(), 10)
    curriculum.add(SimpleDataset(), 10)
    curriculum.add(SimpleDataset(), -1)
    val_dataset = SimpleDataset()
    module = DummyClassificationModule()

    trainer = get_simple_trainer(max_steps=100)
    trainer.fit(module, curriculum, val_dataset)

    assert trainer.state.finished and trainer.state.train_dataloader_idx == len(curriculum)


def test_ratio_curriculum():
    curriculum = RatioCurriculum()
    curriculum.add(SimpleDataset(), 0.1)
    curriculum.add(SimpleDataset(), 0.2)
    curriculum.add(SimpleDataset(), 0.3)
    curriculum.add(SimpleDataset(), 0.4)
    curriculum.add(SimpleDataset(), -1, {"shuffle": False})
    val_dataset = SimpleDataset()
    module = DummyClassificationModule()

    trainer = get_simple_trainer(max_steps=100)
    trainer.fit(module, curriculum, val_dataset)

    assert trainer.state.finished and trainer.state.train_dataloader_idx == len(curriculum)


def test_range_curriculum():
    curriculum = RangeCurriculum()
    curriculum.add(SimpleDataset(), range(10), {"shuffle": False})
    curriculum.add(SimpleDataset(), range(10, 20))
    curriculum.add(SimpleDataset(), range(20, 30))
    curriculum.add(SimpleDataset(), range(30, 40))
    curriculum.add(SimpleDataset(), range(40, 50))
    val_dataset = SimpleDataset()
    module = DummyClassificationModule()

    trainer = get_simple_trainer(max_steps=50)
    trainer.fit(module, curriculum, val_dataset)

    assert trainer.state.finished and trainer.state.train_dataloader_idx == len(curriculum)
