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

import os
from time import sleep

import torch
import torch.nn as nn
from dotenv import load_dotenv

from src.accmt import AcceleratorModule, HyperParameters, Monitor, Optimizer, Scheduler, Trainer, set_seed
from src.accmt.tracker import MLFlow
from src.accmt.utility import RANK

from .dummy_callbacks import DummyCallback
from .dummy_dataset import DummyDataset
from .dummy_metrics import Accuracy, DictMetrics
from .dummy_model import DummyModel


load_dotenv()

# TODO for mlflow, and to log with it, we need to set this environmental variable (mandatory)
MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]


class DummyModule(AcceleratorModule):
    def __init__(self):
        self.model = DummyModel(input_size=2, inner_size=5, output_size=3)
        self.criterion = nn.CrossEntropyLoss()

    def training_step(self, batch):
        sleep(0.1)
        x, y = batch
        x = self.model(x)

        loss = self.criterion(x, y)

        return loss

    def validation_step(self, key, batch):
        sleep(0.1)
        x, y = batch
        x = self.model(x)

        loss = self.criterion(x, y)

        predictions = torch.argmax(x, dim=1)
        references = torch.argmax(y, dim=1)
        extra_references = references.clone()

        if RANK == 0:
            tensors = {"spaa_Latn": torch.tensor([[1, 2, 3, 4]]), "engg_Latn": torch.tensor([[1, 2, 3, 4]])}
        else:
            tensors = {"spa_Latn": torch.tensor([[1, 2, 3, 4]]), "aus_Cyrl": torch.tensor([[1, 2, 3, 4, 5]])}

        return {
            "loss": loss,
            "accuracy": (predictions, references, extra_references),
            "my_own_metric": (predictions, references),
            "test_dict": tensors,
        }


if __name__ == "__main__":
    set_seed(42)

    module = DummyModule()

    train_dataset = DummyDataset()
    val_dataset = DummyDataset()
    val_dataset2 = DummyDataset()

    metrics = [Accuracy("accuracy"), DictMetrics("test_dict")]
    trainer = Trainer(
        hps_config=HyperParameters(
            epochs=2,
            batch_size=(2, 1),
            optimizer=Optimizer.AdamW,
            optim_kwargs={"lr": 0.001, "weight_decay": 0.01},
            scheduler=Scheduler.LinearWithWarmup,
            scheduler_kwargs={"warmup_ratio": 0.03},
        ),
        model_path="dummy_model",
        track_name="Dummy training22",
        run_name="dummy_run",
        evaluate_every_n_steps=4,
        checkpoint_every="eval",
        logging_dir=MLFLOW_TRACKING_URI,
        log_with=MLFlow,
        log_every=2,
        monitor=Monitor(grad_norm=True),
        compile=True,
        eval_when_start=False,
        metrics=metrics,
        callback=DummyCallback(),
        patience=2,
    )

    trainer.log_artifact(".gitignore")
    trainer.register_model_saving("best_valid_loss@0")
    trainer.register_model_saving("best_accuracy/valid_loss@0")
    trainer.register_model_saving("best_accuracy@0@1/valid_loss@0")
    trainer.fit(module, train_dataset, val_dataset)
