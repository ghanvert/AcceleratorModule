# AcceleratorModule
We need to define the logic of our training process. This can be easily done using our **AcceleratorModule** super class.

Here's a simple example:
```python
class ExampleModule(AcceleratorModule):
    def __init__(self):
        self.model = ...

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, status_dict):
        x, y = batch
        # ...
        return train_loss

    def validation_step(self, batch, status_dict):
        x, y = batch

        predictions = ...
        references = ...

        return {
            "loss": val_loss,
            "accuracy": (predictions, references),
            "...": (..., ...)
        }
```

A forward method is not required, although training_step and/or validation_step are mandatory. In the case that your training and validation logic are equal, then you can replace both functions with one: step. AcceleratorModule must have a property self.model derived from nn.Module (PyTorch).

**NOTE**: "status_dict" argument is optional. Only "batch" must be passed.

Here's a detailed information about every method that can be incorporated and customized in your module:
- **def forward(self, x)**: Defines the flow of data and let's you do **self(x)** instead of **self.model(x)**. **Must return torch.Tensor**, which is your model output.
- **def training_step(self, batch)**: Defines the training logic up to the loss value. **Must return torch.Tensor**, which is your scalar loss value.
- **def validation_step(self, batch)**: Defines the validation logic up to the loss value. **Must return torch.Tensor**, which is your scalar loss value NOTE: this function does not require **torch.no_grad()** context manager because it's already done by this library under the hood.
- **def collate_fn(self, batch: list)**: Defines a custom collate function for PyTorch DataLoader. **Must return torch.Tensor**, which is your stack of tensors containing a batch of information.
- **def get_optimizer(self)**: Defines a custom PyTorch optimizer. **Must return the optimizer itself**.
- **def get_scheduler(self, optimizer, steps_per_epoch: int, epochs: int)**: Defines a custom PyTorch scheduler. **Must return the scheduler itself**.
- **def get_train_dataloader(self)**: Defines a custom PyTorch DataLoader class for training **Must return a PyTorch DataLoader**.
- **def get_validation_dataloader(self)**: Defines a custom PyTorch DataLoader class for validation. **Must return a PyTorch DataLoader**.

You can get the total number of parameters of your main model via the **len** function:
```python
total_params = len(module)
```

## But, what is "status_dict"?
**status_dict** is a dictionary containing relevant information about your training process, and it's required as parameter in the methods **training_step**, **validation_step** and **step**. Contains the following keys:
- **best_train_loss** (float): Best train loss calculated. Updated at the end of model evaluation.
- **best_valid_loss** (float): Best validation loss calculated. Updated at the end of model evaluation.
- **epoch** (int): Current epoch. Updated at the beggining of every epoch.
- **epoch_step** (int): Current step on your epoch. Updated every pass of your train logic.
- **global_step** (int): Global step of your whole training process. Updated every pass of your train logic.
- **eval_global_step** (int): Evaluation global step of your validation process. Updated every pass of your validation logic.
- **evaluations_done** (int): How many evaluations have been done.
- **additional_metrics** (dict): Dictionary containing every additional metric value added.

If you're checkpointing every N defined steps, there will be an extra key:
- **skip_batches** (int): How many batches are meant to be skipped since last checkpoint.

## "self.model" and "self.teacher"
**model** and **teacher** in the subclass from AcceleratorModule are special keywords that this library will use under the hood. We support teacher-student approach to train these kind of models, so **teacher** is also reserved (in this cases, **model** will be the student).
