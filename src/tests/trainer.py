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

import pytest
from dotenv import load_dotenv
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.accmt import DataCollatorForSeq2Seq, HyperParameters, Scheduler, Trainer

from .dummy_callbacks import DummyCallback
from .dummy_datasets import TranslationDataset
from .dummy_modules import DummyTranslationExtendedModule, DummyTranslationModule


@pytest.fixture
def setup_training():
    load_dotenv()

    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    max_length = 128
    train_dataset = TranslationDataset(tokenizer, max_length=max_length, is_val=False)
    val_dataset = TranslationDataset(tokenizer, max_length=max_length, is_val=True)

    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

    training_module = DummyTranslationModule(model, tokenizer)
    training_extended_module = DummyTranslationExtendedModule(model, tokenizer)

    trainer = Trainer(
        hps_config=HyperParameters(
            epochs=2,
            batch_size=(4, 8),
            optimizer="AdamW",
            optim_kwargs={"lr": 1e-4},
            scheduler=Scheduler.CosineWithWarmup,
            scheduler_kwargs={"warmup_ratio": 0.1},
            step_scheduler_per_epoch=False,
        ),
        model_path="test_model",
        track_name="accmt-tests",
        run_name="accmt-translation-test",
        patience=-1,
        evaluate_every_n_steps=100,
        checkpoint_every="epoch",
        logging_dir=MLFLOW_TRACKING_URI,
        log_with="mlflow",
        log_every=2,  # should log every 4 steps (2 batches * 2 grad accumulation steps)
        grad_accumulation_steps=2,
        gradient_checkpointing=True,
        collate_fn=DataCollatorForSeq2Seq(tokenizer),
        compile=True,
        eval_when_start=True,
        cleanup_cache_every_n_steps=50,
        callbacks=[DummyCallback(), DummyCallback()],
    )

    val_datasets = {"val1": val_dataset, "val2": val_dataset}

    trainer.register_model_saving("bleu@val1")
    trainer.register_model_saving("bleu@val2")
    trainer.register_model_saving("bleu@val1@val2")
    trainer.register_model_saving("bleu")
    trainer.log_artifact("artifact_test.yml")
    trainer.log_artifacts("artifact_test")

    return trainer, training_module, training_extended_module, train_dataset, val_datasets


def test_training_completes(setup_training):
    trainer, training_module, _, train_dataset, val_datasets = setup_training

    trainer.fit(training_module, train_dataset, val_datasets)

    assert trainer.state.finished
    assert training_module.state.finished


def test_training_extended_completes(setup_training):
    trainer, _, training_extended_module, train_dataset, val_datasets = setup_training

    trainer.fit(training_extended_module, train_dataset, val_datasets)

    assert trainer.state.finished
    assert training_extended_module.state.finished
