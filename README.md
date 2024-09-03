# AcceleratorModule
Module based on Accelerate 🤗 for distributed training accross multiple GPUs, with focus on readability and ease to customize experiments. We also integrate modified versions of DataCollators from Transformers library for huggingface standard tokenizers to integrate with different environments.

NOTE: Some features might not be tested and could cause problems. Feel free to open an issue or send a PR to fix any problem found.

AcceleratorModule will take care of the heavy lifting of distributed training on many GPUs. Accelerate is quite simple, and it has many adventages over PyTorch Lightning, mainly because it doesn't abstract the low level part of the training loop, so you can customize it however you want. The main idea of this little project is to have a standard way to make distributed training. This module let's you:
- Define the logic involved for training, validation and test.
- Define the logic involved to calculate different metrics in a simple and reduced manner.
- Save checkpoints to recover training progress.
- Early stopping by evaluating any best average metric.
- Define the hyperparameters in a simple YAML file or HyperParameters object.
- Visualize training progress using any supported tracker.
- Manipulate how often are checkpoints done, evaluations, logging, model saving, etc.
- Easily set an experimental environment by calling **set_seed** function.
- Train **transformers** standard models with a few lines of code.
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

The AcceleratorModule class has 3 main methods:
- **forward**: Defines the flow of data (completely optional, since you can directly call 'self.model').
- **training_step**: Defines the training logic up to the loss function.
- **validation_step**: Defines the validation logic up to the loss function.

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
        return val_loss

    # if you want to calculate metrics on a test dataset, you can do the following:
    def test_step(self, batch):
        x, y = batch
        # ...
        predictions = ...
        references = ...

        return {
            "accuracy": (predictions, references),
            "any_other_metric": (predictions, references)
        }
```

More information about module structure [here](https://github.com/ghanvert/AcceleratorModule/blob/main/docs/module_structure.md).

To train this Module, you need a **Trainer** class:
```python
from accmt import Trainer, HyperParameters

trainer = Trainer(
    #hps_config="hps_config.yaml",  # <--- can also be a YAML file.
    hps_config=HyperParameters(epochs=2),
    model_path="checkpoint_folder"
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
  optim:
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
accmt launch train.py -N=8 --strat=deepspeed-2-bf16
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

You can also save model in 3 different modes:
- **best_valid_loss**: Saves the model whenever the validation loss is the best.
- **best_train_loss**: Saves the model whenever the train loss is the best.
- **always**: Save the model everytime it's possible.

Or the following format:
- **best_{METRIC}**: If you're using an specific metric to save the model, specify it after 'best_'. (e.g. 'best_accuracy')

And you can activate movel saving below or above a specific metric (e.g. if specified **best_valid_loss**, then model will be saved when validation loss is below or above the specified thresholds).

```python
trainer = Trainer(
    # ... Other parameters.
    model_path="model", # Path where to save model.
    model_saving="best_valid_loss", # Model saving mode.
    model_saving_below=0.67 # Save model below this threshold (e.g. below 0.67 validation loss).
    model_saving_above=0.01 # Completely optional.
)
```


## Gradient Accumulation
When training big models, size in memory becomes a huge problem. One way to avoid that is to not always step the optimizer, instead accumulate gradients for a certain amount of steps. This is very easy to do, just configure the parameter **grad_accumulation_steps** for the amount of steps you want to accumulate gradients before stepping.


## Logging training progress
Logging training progress is set by default in ACCMT, as it is essential to track how good our experiments are, and determine if we're good to pause training.

There are only 2 paremeters to change for this (in the Trainer constructor):
- **logging_dir**: Specifies a logging dir (default is "logs"). This can be a directory path or a URL.
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

At the moment, there are 3 collators directly inspired on **transformers** library (with a little bit of modifications):
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

        self.teacher.eval() # set teacher to evaluation mode
```

During training, the teacher model will only provide outputs, and will not have its parameters updated.

**NOTE**: In order to successfully load models into hardware, we must use **self.teacher** for teacher model, and **self.model** for student model.

If using KL Divergence approach for the loss function, our **step** method will look something like this:
```python
import torch
import torch.nn.functional as F
# other imports...

# other logic for module...

def step(self, batch):
    x = batch
    with torch.no_grad(): # no gradients required for teacher model
        teacher_logits = self.teacher(**x).logits

    student_output = self.model(**x)
    student_logits = student_output.logits

    soft_prob = F.log_softmax(student_logits / self.T, dim=-1)
    soft_targets = F.softmax(teacher_logits / self.T, dim=-1)

    kd_loss = F.kl_div(soft_prob, soft_targets, reduction="batchmean") * (self.T**2)
    loss = self.alpha * student_output.loss + (1. - self.alpha) * kd_loss

    return loss
```


## Notes
I will continue to update this repository to add more features overtime. If you want to contribute to this little project, feel free to make a PR 🤗.
