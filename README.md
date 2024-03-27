# AcceleratorModule
Module similar to Lightning Module for distributed training, but with Accelerator 🤗

## Module Structure
Import Accelerator 🤗 and AcceleratorModule:
```python
from accelerate import Accelerator
from trainer import AcceleratorModule
```

The AcceleratorModule class has 3 main methods:
- **forward**: Defines the flow of data.
- **training_step**: Defines the training logic up to the loss function.
- **validation_step**: Defines the validation logic up to the loss function.

The structure looks like this:
```python
class ExampleModule(AcceleratorModule):
    def __init__(self):
        self.model = ...

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        # ...
        return train_loss

    def validation_step(self, batch):
        x, y = batch
        # ...
        return val_loss
```

A **forward** method is not required, although **training_step** and/or **validation_step** are mandatory.
**AcceleratorModule** must have a property **self.model** derived from nn.Module (PyTorch).

To train this Module, you need a **Trainer** class:
```python
from trainer import Trainer

trainer = Trainer(
    accelerator,
    hps_file_config="hps_config.yaml",
    checkpoint="checkpoint_folder"
)
```

Finally, we can train our model by using the **.fit()** function, providing our AcceleratorModule and the train and validation datasets (from PyTorch):
```python
trainer.fit(module, train_dataset, val_dataset)
```

## Run
To run training, we use Accelerator 🤗 in the background, so you must use the corresponding CLI:
```python
accelerate launch train.py
```

You can use any Accelerate configuration that you want 🤗 (DDP, FSDP or DeepSpeed).
