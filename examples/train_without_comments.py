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
import torch.nn as nn
from dummy_dataset import DummyDataset

from accmt import AcceleratorModule, HyperParameters, Monitor, Optimizer, Scheduler, Trainer, accelerator, set_seed
from accmt.tracker import MLFlow

from .dummy_model import DummyModel


set_seed(42)


class DummyModule(AcceleratorModule):
    def __init__(self):
        self.model = DummyModel(input_size=2, inner_size=5, output_size=3)
        self.criterion = nn.CrossEntropyLoss()

    def training_step(self, batch):
        x, y = batch
        x = self.model(x)

        loss = self.criterion(x, y)

        return loss

    def validation_step(self, batch):
        x, y = batch
        x = self.model(x)

        loss = self.criterion(x, y)

        predictions = torch.argmax(x, dim=1)
        references = torch.argmax(y, dim=1)

        return {"loss": loss, "accuracy": (predictions, references)}


module = DummyModule()

train_dataset = DummyDataset()
val_dataset = DummyDataset()

trainer = Trainer(
    hps_config=HyperParameters(
        epochs=2,
        batch_size=(2, 1, 1),
        optimizer=Optimizer.AdamW,
        optim_kwargs={"lr": 0.001, "weight_decay": 0.01},
        scheduler=Scheduler.LinearWithWarmup,
        scheduler_kwargs={"warmup_ratio": 0.03},
    ),
    model_path="dummy_model",
    track_name="Dummy training",
    run_name="dummy_run",
    resume=False,
    model_saving="best_accuracy",
    evaluate_every_n_steps=1,
    checkpoint_every="eval",
    logging_dir="localhost:5075",
    log_with=MLFlow,
    log_every=2,
    monitor=Monitor(grad_norm=True),
    additional_metrics=["accuracy"],
    compile=True,
    dataloader_num_workers=accelerator.num_processes,
    eval_when_start=True,
)

trainer.fit(module, train_dataset, val_dataset)
