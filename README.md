# AcceleratorModule
Module based on Accelerate 🤗 for distributed training accross multiple GPUs, with focus on readability and ease to customize experiments. We also integrate modified versions of DataCollators from Transformers library for huggingface standard tokenizers to integrate with different environments.

NOTE: Some features might not be tested and could cause problems. Feel free to open an issue or send a PR to fix any problem found.

AcceleratorModule will take care of the heavy lifting of distributed training on many GPUs. Accelerate is quite simple, and it has many adventages over PyTorch Lightning, mainly because it doesn't abstract the low level part of the training loop, so you can customize it however you want. The main idea of this little project is to have a standard way to make distributed training. This module let's you:
- Define the logic involved for training and validation.
- Define the logic involved to calculate different metrics in a simple and reduced manner.
- Save checkpoints to recover training progress.
- Early stopping by evaluating any best average metric.
- Define the hyperparameters in a simple YAML file or HyperParameters object.
- Visualize training progress using any supported tracker.
- Manipulate how often are checkpoints done, evaluations, logging, model saving, etc.
- Easily set an experimental environment by calling **set_seed** function.
- And more.

## Installation
AcceleratorModule is available via pip:
```bash
pip install accmt
```


## Module Structure
Import AcceleratorModule:
```python
from accmt import AcceleratorModule
```

The AcceleratorModule class has 2 main methods:
- **training_step**: Defines the training logic.
- **validation_step**: Defines the validation logic.

The structure looks like this:
```python
class ExampleModule(AcceleratorModule):
    def __init__(self):
        self.model = ...

    def training_step(self, batch):
        x, y = batch
        # ...
        return train_loss

    def validation_step(self, batch):
        x, y = batch
        # ...
        return {
            "loss": val_loss,
            # any other metric...
        }
```

More information about module structure [here](https://github.com/ghanvert/AcceleratorModule/blob/main/docs/module_structure.md).

To train this Module, you need a **Trainer** class:
```python
from accmt import Trainer, HyperParameters

trainer = Trainer(
    #hps_config="hps_config.yaml",  # <--- can also be a YAML file.
    hps_config=HyperParameters(epochs=2),
    model_path="model_folder"
    # ... other arguments
)
```

More information about trainer [here](https://github.com/ghanvert/AcceleratorModule/blob/main/docs/trainer.md).

## HPS config file
This is a YAML file containing hyperparameters for your training. The structure looks like the following:
```yaml
hps:
  epochs: 40
  batch_size: 35
  optimizer:
    type: AdamW
    lr: 1e-3
    weight_decay: 1e-3
  scheduler:
    type: OneCycleLR
    max_lr: 1e-3
```

An optimizer (**optim**) is necessary, while a **scheduler** is optional (do not specify if you don't want to).

Available optimizer types are the following:
|  Optimizer  |  Source      |
|-------------|--------------|
|  [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) |  PyTorch |
|  [Adadelta](https://pytorch.org/docs/stable/generated/torch.optim.Adadelta.html#torch.optim.Adadelta) |  PyTorch |
|  [Adagrad](https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html#torch.optim.Adagrad) |  PyTorch |
|  [Adamax](https://pytorch.org/docs/stable/generated/torch.optim.Adamax.html#torch.optim.Adamax) |  PyTorch |
|  [AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW) |  PyTorch |
|  [Adafactor](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.Adafactor) |  HuggingFace |
|  [ASGD](https://pytorch.org/docs/stable/generated/torch.optim.ASGD.html#torch.optim.ASGD) |  PyTorch |
|  [LBFGS](https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html#torch.optim.LBFGS) |  PyTorch |
|  [NAdam](https://pytorch.org/docs/stable/generated/torch.optim.NAdam.html#torch.optim.NAdam) |  PyTorch |
|  [RAdam](https://pytorch.org/docs/stable/generated/torch.optim.RAdam.html#torch.optim.RAdam) |  PyTorch |
|  [RMSprop](https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html#torch.optim.RMSprop) |  PyTorch |
|  [Rprop](https://pytorch.org/docs/stable/generated/torch.optim.Rprop.html#torch.optim.Rprop) |  PyTorch |
|  [SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD) |  PyTorch |
|  [SparseAdam](https://pytorch.org/docs/stable/generated/torch.optim.SparseAdam.html#torch.optim.SparseAdam) |  PyTorch |


Available schedulers types are the following:
|  Scheduler                        |  Source      |
|-----------------------------------|--------------|
|  [StepLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LinearLR.html#torch.optim.lr_scheduler.LinearLR) |  PyTorch |
|  [LinearLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LinearLR.html#torch.optim.lr_scheduler.LinearLR) |  PyTorch |
|  [ExponentialLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html#torch.optim.lr_scheduler.ExponentialLR)|  PyTorch |
|  [CosineAnnealingLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html#torch.optim.lr_scheduler.CosineAnnealingLR)|  PyTorch |
|  [CyclicLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CyclicLR.html#torch.optim.lr_scheduler.CyclicLR) |  PyTorch |
|  [OneCycleLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html#torch.optim.lr_scheduler.OneCycleLR)                       |  PyTorch |
|  [CosineAnnealingWarmRestarts](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts) |  PyTorch |
|  [CosineWithWarmup](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.get_cosine_schedule_with_warmup) |  HuggingFace |
|  [Constant](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.get_constant_schedule) |  HuggingFace |
|  [ConstantWithWarmup](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.get_constant_schedule_with_warmup) |  HuggingFace |
|  [CosineWithHardRestartsWithWarmup](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.get_cosine_with_hard_restarts_schedule_with_warmup) |  HuggingFace |
|  [InverseSQRT](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.get_inverse_sqrt_schedule) |  HuggingFace |
|  [LinearWithWarmup](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.get_linear_schedule_with_warmup) |  HuggingFace |
|  [PolynomialDecayWithWarmup](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.get_polynomial_decay_schedule_with_warmup) |  HuggingFace |

Finally, we can train our model by using the **.fit()** function, providing our AcceleratorModule and the train and validation datasets (from PyTorch):
```python
trainer.fit(module, train_dataset, val_dataset)
```

More information about HPS config file [here](https://github.com/ghanvert/AcceleratorModule/blob/main/docs/hps_file_config.md).

## Run
To run training, you can use **accmt** command-line utilities (which is a wrapper around Accelerate 🤗)
```bash
accmt launch -N=8 --strat=deepspeed-2-bf16 train.py
```
This will run on 8 GPUs with DeepSpeed zero stage 2, with a mixed precision of bfloat16. If **-N** argument is not specified, **accmt** will launch N numbers of processes, where N will be equal to the number of GPUs detected in your system. Also, if **--strat** is not specified, default strategy will be **DDP** with no mixed precision.

You can use any Accelerate configuration that you want 🤗 (DDP, FSDP or DeepSpeed). For more strategies, check:
```bash
accmt strats  # --ddp | --fsdp | --deepspeed    <--- optional filters.
```

**NOTE**: You can also use **accelerate** command-line utilities instead.

More information about command-line utilities [here](https://github.com/ghanvert/accmt-cli).

## Checkpointing
Checkpointing is a default process in ACCMT, and it's customizable with some parameters in the Trainer constructor:
```python
trainer = Trainer(
    # ... Other parameters.
    checkpoint_every="2ep", # Checkpoint every N epochs, in this case, every 2 epochs.
    resume=True # Whether you want to resume from checkpoint (True), or start from scratch (False).
    # if not specified (None), resuming will be done automatically.
)
```


## Save model
Model saving is an integrated feature of ACCMT. You can enable it by specifying a directory where to save the model.

You can also save model in 2 different default modes:
- **best_valid_loss**: Saves the model whenever the validation loss is the best (default if not specified).
- **best_train_loss**: Saves the model whenever the train loss is the best.

Or the following format:
- **best_{METRIC}**: If you're using an specific metric to save the model, specify it after 'best_'. (e.g. 'best_accuracy'). **NOTE**: 'best_' prefix is optional.

And you can activate movel saving below or above a specific metric.

```python
trainer = Trainer(...)
trainer.register_model_saving("accuracy", saving_above=0.2)
```


## Gradient Accumulation
When training models, larger batch sizes are often more stable than little ones, but it comes at a cost of VRAM. One way to avoid this is to accumulate gradients for N steps. This way, we simulate larger batch sizes without increasing VRAM usage.
```python
trainer = Trainer(..., grad_accumulation_steps=2)
```


## Logging training progress
Logging training progress is set by default in ACCMT, as it is essential to track how good our experiments are, and determine if we're good to pause training.

There are only 2 parameters to change for this (in the Trainer constructor):
- **track_with**: Specify the tracker you want to use. Only available option (for now) is "mlflow".
- **logging_dir**: Specifies a logging directory (default is "logs"). This can be a directory path or a URL.
- **log_every**: Log every N number of steps (default is 1).


## Collate Functions
You can implement your own collate function by overriding **collate_fn** from AcceleratorModule:
```python
class ExampleModule(AcceleratorModule):
    # Rest of the code...

    def collate_fn(self, batch: list):
        # Your collate function logic here.

        return batch # Output taken in training and validation steps.
```

There is another and simplier way to add collators that I'm going to be building in the future, and that is using a specific **DataCollator** built into this library.

At the moment, there are 3 collators directly inspired on the **transformers** library (with some modifications like recursive approaches to iterate over different arguments in the `__getitem__` function of the Dataset):
- **DataCollatorForSeq2Seq**: Adds efficient padding when dealing with sequence-to-sequence problems.
- **DataCollatorForLongestSequence**: Adds efficient padding for a batch.
- **DataCollatorForLanguageModeling**: Implements Masked Language Modeling (MLM) task.

Example:
```python
from accmt import Trainer, DataCollatorForSeq2Seq

tokenizer = ... # a tokenizer from 'transformers' library.

trainer = Trainer(
    hps_config="hps_config.yaml",
    model_path="dummy_model",
    collate_fn=DataCollatorForSeq2Seq(tokenizer)
)
```


## Teacher-Student support
A Teacher-Student approach let's you mimic the behaviour of a bigger model (teacher) in a smaller model (student). This is a method for model distillation, useful to save computational resources and accelerate inference.

To load teacher and student models, we can do the following in the module constructor:
```python
class TeacherStudentExampleModule(AcceleratorModule):
    def __init__(self):
        self.teacher = ... # teacher model
        self.model = ...   # student model
```

During training, the teacher model will only provide outputs, and will not have its parameters updated.

**NOTE**: In order to successfully load models into hardware, we must use **self.teacher** for teacher model, and **self.model** for student model.


## Notes
I will continue to update this repository to add more features overtime. If you want to contribute to this little project, feel free to make a PR 🤗.
