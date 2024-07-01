# YAML Hyperparameters
This is a YAML file format to specify different hyperparameters for your training.

Here's a list of hyperparameters:
- **epochs** (required): Number of epochs to train.
- **batch_size** (required): Number of batches to be processed in parallel.
- **optim** (required): Optimizer section.
    - **type**: Name of the optimizer supported by this library.
    - **"..."**: You can specify in a list the parameters required by the optimizer, such as "lr", "weight_decay", etc.
- **scheduler** (*optional*): Scheduler section.
    - **type**: Name of the optimizer supported by this library.
    - **"...**: You can specify in a list the parameters required by the scheduler, such as "num_warmup_steps".

## Example
Here's an example on how to write an HPS file:
```yaml
hps:
  epochs: 1
  batch_size: 32
  optim:
    type: AdamW
    lr: 1e-6
    weight_decay: 0.01
    eps: 1e-6
    betas: [0.9, 0.98]
  scheduler:
    type: LinearWithWarmup
    num_warmup_steps: 29770
```

If you want a constant learning rate, you can remove the **scheduler** section.

**NOTE**: It is possible to use string numbers such as "1e-6" (scientific notation) and lists to represent tuples.

## Information about arguments from Scheduler section
Some schedulers asks you to specify some extra information that derives from the length of the dataloader or the number of steps per epoch. These arguments will automatically be handled by this library, so you do not have to specify or calculate them. Here's a list of these arguments that are required by some schedulers:
- **last_epoch**
- **steps_per_epoch**
- **num_training_steps**
- **epochs**

If you wrote an argument that does not correspond with the respective scheduler, it will be ignored.
