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
from dummy_metric import Accuracy
from dummy_model import DummyModel

# accmt import should be at the beggining of your script, since internal Accelerator object is initialized here.
from accmt import AcceleratorModule, HyperParameters, Monitor, Optimizer, Scheduler, Trainer, accelerator, set_seed


# For experimental setups you might want your model weights to always be the same.
# Also you might want any random buffer or state to be the same in every experiment.
# Consider using 'set_seed' helper function to do so, at the beggining of your Python script.
set_seed(42)

# Objects like tokenizer are recommended to be initialized outside of the module, since the might be
# required not only by the module, but for datasets and Trainer arguments.


class DummyModule(AcceleratorModule):
    def __init__(self):
        # The following attributes are reserved for AcceleratorModule:
        #   - 'model': Holds your model as nn.Module class. Must be declared.
        #   - 'teacher': Holds your teacher model as nn.Module class. Declare it only if you need it.
        #   - 'state': Dictionary containing information about training process.

        self.model = DummyModel(input_size=2, inner_size=5, output_size=3)
        self.criterion = nn.CrossEntropyLoss()

    def training_step(self, batch):
        # since our dataset logic returns input and target, we decompose batch into inputs (x) and targets (y).
        x, y = batch
        x = self.model(x)

        loss = self.criterion(x, y)

        # loss needs to be an scalar tensor (just one value)
        return loss

    # 'validation_step' is a different structure compared to train and validation logics. This function is
    # intenteded to be used whenever we want to calculate metrics such as accuracy, bleu, f1, etc.
    def validation_step(self, key, batch):
        x, y = batch
        x = self.model(x)

        loss = self.criterion(x, y)

        predictions = torch.argmax(x, dim=1)
        references = torch.argmax(y, dim=1)

        # we must return a dictionary containing a key, which indicates what metric is, and a
        # value indicating a tuple where the first element are the predictions as batch, and references
        # as the second element, representing the actual targets as batch. The arguments will depend on the 'compute'
        # function defined in your `Metric` module.
        return {
            "loss": loss,  # this is MANDATORY key-value
            "accuracy": (predictions, references),
        }


module = DummyModule()

# We need to define our datasets. 'train_dataset' is mandatory and 'val_dataset' is for evaluation (optional).
# Here, both datasets are the same just for simplicity. In a real training setting you would want
# these datasets to be different.
train_dataset = DummyDataset()
val_dataset = DummyDataset()

# Trainer class contains all the arguments to control model saving, checkpointing, logging, optimization, etc.
trainer = Trainer(
    hps_config=HyperParameters(
        epochs=2,  # how many times we run over the train dataset.
        # max_steps=10,  # it is also possible to specify a maximum number of steps instead of epochs.
        batch_size=(2, 1),
        # 'batch_size' can be an integer value or tuple, where elements are: (train_batch_size, val_batch_size).
        # Use an integer value to set the batch size equally for both sets.
        optimizer=Optimizer.AdamW,
        optim_kwargs={"lr": 0.001, "weight_decay": 0.01},  # Optimizer arguments as dictionary
        scheduler=Scheduler.LinearWithWarmup,
        scheduler_kwargs={"warmup_ratio": 0.03},  # Scheduler arguments as dictionary
        # Note that 'warmup_ratio' is a special argument from this library for schedulers that
        # internally require 'num_warmup_steps'.
    ),
    # hps_config="hps_example.yaml"    <=== this is also valid
    model_path="dummy_model",  # we must specify the model's folder path.
    # The checkpoint would be another folder containing the last model checkpoint. If not specified,
    # its name will be equal to 'checkpoint-{model_path}', in this case: checkpoint-dummy_model.
    # If you want to change the name of its folder, you can use the 'checkpoint' argument:
    #     checkpoint = "my_dummy_checkpoint"
    track_name="Dummy training",  # Track name is useful when logging, so you can have some runs on this track.
    run_name="dummy_run",  # 'run_name' is a specific argument for MLFlow tracker (and others).
    resume=False,
    # if you specify 'resume', you will force the training process to resume or not training.
    # When not specified, it will detect if a checkpoint exists and resume from there
    # by default (when not specified), we save the model based on the validation loss ('best_valid_loss').
    evaluate_every_n_steps=1,
    # by default, evaluation occurs every epoch. Use this argument to do evaluation every N steps.
    checkpoint_every="eval",  # you can checkpoint every N epochs, evaluations or steps. See docs for formatting.
    logging_dir="localhost:5075",
    # this can also be a folder name (local). If it's an URL-like, logging will ocurr on the corresponding server.
    log_with="mlflow",  # TODO: only "mlflow" is supported for now.
    log_every=2,  # log every N steps. You might want to consider increasing this value to optimize training time.
    monitor=Monitor(grad_norm=True),  # you can enable or disable monitoring for certain metrics.
    metrics=Accuracy("accuracy", greater_is_better=True),  # you can also pass a list of metric modules.
    compile=True,  # compiles the model for better perfomance. If you need to manually compile the model, you can
    # disable this and do it manually in the `AcceleratorModule`.
    gradient_checkpointing=True,  # in case that model has a `gradient_checkpointing_enable` function.
    dataloader_num_workers=accelerator.num_processes,  # optimized setup for internal DataLoaders.
    eval_when_start=True,  # evaluate model at the very beggining, before training (default value is 'False').
)
# specify how to save the model. Correct format is: 'best_{METRIC}' or '{METRIC}'.
trainer.register_model_saving("best_accuracy", saving_above=0.2)

trainer.fit(module, train_dataset, val_dataset)  # dataloaders are internally constructed

# You can run training by typing:
#
#       accmt launch -N=8 --strat=deepspeed-2-bf16 train.py
#
#   That will run the training with 8 processes using DeepSpeed 2 with a mixed precision of bfloat16.
#   See 'accmt strats' to check all available configurations. You can also do 'accmt strats --ddp', '--fsdp' or
#   '--deepspeed' to filter strategies.
#
#   '--strat' can also be a config file from Accelerate. To further customize a configuration file, you can type:
#
#       accelerate config --config_file=your_configuration.yaml
#
#   then you will need to answer all the questions asked.
#   If you do not specify '--strat', default strategy will be DistributedDataParallel (DDP).
