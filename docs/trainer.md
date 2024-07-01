# Trainer
The **Trainer** class defines many arguments to customize your training process. Here's a detailed information about the constructor of this class:

- **hps_file_config** (str): YAML hyperparameters file path.
- **checkpoint** (str, *optional*, defaults to "checkpoint1"): Checkpoint path.
- **resume** (bool, *optional*, defaults to False): Whether to resume from checkpoint.
- **model_path** (str, *optional*, defaults to None): Path to save model. If not specified, it will name the model path based on the **hps_file_config** filename (without the extension).
- **model_saving** (str, *optional*, defaults to "best_valid_loss"): Type of model saving. It can be one of the following values:
    - **"best_valid_loss"**: Saves the model whenever the validation loss is the best recorded.
    - **"best_train_loss"**: Saves the model whenever the training loss is the best recorded.
    - **"always"**: Saves the model always at the end of every evaluation.
- **evaluate_every_n_steps** (int, *optional*, defaults to None): Evaluate model in validation dataset (if implemented) every N steps. If this is set to None (default option), evaluation will happen at the end of every epoch.
- **enable_checkpointing** (bool, *optional*, defaults to True): Whether to save checkpoints.
- **checkpoint_strat** (str, *optional*, defaults to "epoch"): Strategy to save checkpoint. It can be one of the following values:
    - **"epoch"**: Save a checkpoint at the end of every epoch, every N epochs specified in **checkpoint_every** parameter.
    - **"step"**: Save a checkpoint every N steps specified in **checkpoint_every** parameter.
    - **"eval"**: Save a checkpoint after every evaluation done.
- **checkpoint_every** (int, *optional*, defaults to 1): Checkpoint every N steps or epochs (determined by "checkpoint_strat"). If **checkpoint_strat** is set to "eval", this parameter is not considered.
- **logging_dir** (str, *optional*, defaults to "logs"): Path where to save logs to show progress.
- **log_with** (str, *optional*, defaults to accmt.TensorBoard): Logger to log metrics.
- **log_every** (int, *optional*, defaults to 1): Log every N steps.
- **grad_accumulation_steps** (int, *optional*, defaults to None): Accumulate gradients for N steps. Useful to simulate large batches when memory is not enough. If set to None or 1, no accumulation will be performed.
- **set_to_none** (bool, *optional*, defaults to True): From PyTorch documentation: "instead of setting to zero, set the grads to None. This will in general have lower memory footprint, and can modestly improve performance". Some optimizers have a different behaviour if the gradient is 0 or None. See PyTorch docs for more information: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
- **shuffle_train** (bool, *optional*, defaults to True): Whether to shuffle train DataLoader.
- **shuffle_validation** (bool, *optional*, defaults to False): Whether to shuffle validation DataLoader.
- **model_saving_below_loss** (float, *optional*, defaults to float("inf")): Start saving model on this loss (based on **model_saving**). Default is always.
- **collate_fn** (function or list of collate functions, *optional*, defaults to None): Collate function to be implemented in dataloaders. If module overrides **collate_fn** from AcceleratorModule class, then that function will be used instead of the one specified on this constructor. If a list of collate functions is given, then every collate function will affect the batch in the given order.
- **max_shard_size** (str, *optional*, defaults to "10GB"): Max model shard size to be used.
- **safe_serialization** (bool, *optional*, defaults to False): Whetherto save model using safe tensors or the traditional PyTorch way. If True, some tensors will be lost.
- **optimizations** (list, *optional*, defaults to None): Optimizations from accmt.optimizations to be applied during training.
- **fused** (bool, *optional*, defaults to True): Whether to use fused optimizer when available.
- **compile** (bool, *optional*, defaults to False): Whether to call torch.compile on model.
- **train_loss_metric_name** (str, *optional*, defaults to "train_loss"): Metric name for train loss in logs.
- **val_loss_metric_name** (str, *optional*, defaults to "val_loss"): Metric name for validation loss in logs.

## Train
To train a model, we must have initialized our Trainer. Then, we can call **fit** function. It requires the following arguments:
- **module** (AcceleratorModule): AcceleratorModule containing the training logic.
- **train_dataset** (Dataset, *optional*, defaults to None): Dataset class from PyTorch containing the train dataset logic.
- **val_dataset** (Dataset, *optional*, defaults to None): Dataset class from PyTorch containing the validation dataset logic.

**train_dataset** and **val_dataset** are None by default because you can define the logic to get the DataLoader using **get_train_dataloader** and **get_validation_dataloader** in the AcceleratorModule subclass. If those are not defined, then the DataLoader will be created using an standard method.
