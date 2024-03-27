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

...
