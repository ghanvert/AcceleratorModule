# Trainer
The **Trainer** class defines many arguments to customize your training process. Here's a detailed information about the constructor of this class:

- **hps_config** (`str`, `dict` or `HyperParameters`): YAML or JSON file path, dictionary with hyperparameters or HyperParameters class.
- **model_path** (`str`): Path to save model.
- **track_name** (`str`, *optional*, defaults to `None`): Track name for trackers. If set to `None` (default), the track name will be the model's folder name.
- **checkpoint** (`str`, *optional*, defaults to `None`): Checkpoint path name. If not defined (`None`), then checkpoint path will be the same as **model_path** with a prefix "checkpoint-".
- **resume** (`bool`, *optional*, defaults to `False`): Whether to resume from checkpoint.
- **model_saving** (`str`, *optional*, defaults to `"best_valid_loss"`): Type of model saving. It can be one of the following values:
    - **"best_valid_loss"**: Saves the model whenever the validation loss is the best recorded.
    - **"best_train_loss"**: Saves the model whenever the training loss is the best recorded.
    - **"always"**: Saves the model always at the end of every evaluation.
    - **None**: If not specified, model saving will be disabled.
- **evaluate_every_n_steps** (`int`, *optional*, defaults to `None`): Evaluate model in validation dataset (if implemented) every N steps. If this is set to `None` (default option), evaluation will happen at the end of every epoch.
- **checkpoint_every** (`str`, *optional*, defaults to `"epoch"`): Checkpoint every N steps, epochs or evaluations (e.g. "2 epochs", "eval", "1eval", "3 evaluations", "50 steps", "30ep", etc). If not specified (None), checkpointing will be disabled.
- **logging_dir** (`str`, *optional*, defaults to `"logs"`): Path where to save logs to show progress. This can also be a URL related to **log_with** service.
- **log_with** (`accmt.tracker`, *optional*, defaults to `None`): Tracker to log metrics. Available options are imported from accmt: TensorBoard, WandB, CometML, Aim, MLFlow, ClearML and DVCLive. 
- **log_every** (`int`, *optional*, defaults to `1`): Log every N steps.
- **grad_accumulation_steps** (`int`, *optional*, defaults to `None`): Accumulate gradients for N steps. Useful for training large models and simulate large batches when memory is not enough. If set to `None` or `1`, no accumulation will be performed.
- **clip_grad** (`float`, *optional*, defaults to `None`): Performs gradient clipping in between backpropagation and optimizer's step function.
- **set_to_none** (`bool`, *optional*, defaults to `True`): From PyTorch documentation: "instead of setting to zero, set the grads to None. This will in general have lower memory footprint, and can modestly improve performance". Some optimizers have a different behaviour if the gradient is 0 or None. See PyTorch docs for more information: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
- **shuffle_train** (`bool`, *optional*, defaults to `True`): Whether to shuffle train DataLoader.
- **shuffle_validation** (`bool`, *optional*, defaults to `False`): Whether to shuffle validation DataLoader.
- **sampler** (`list` or `Any`, *optional*, defaults to `None`): Sampler (or list of samplers) for train DataLoader.
- **model_saving_below** (`float`, *optional*, defaults to `None`): Start saving the model below this metric value (based on **model_saving**).
- **model_saving_aboce** (`float`, *optional*, defaults to `None`): Start saving the model above this metric value (based on **model_saving**).
- **collate_fn** (`function` or `list` of collate functions, *optional*, defaults to `None`): Collate function to be implemented in dataloaders. If module overrides **collate_fn** from AcceleratorModule class, then that function will be used instead of the one specified on this constructor. If a list of collate functions is given, then every collate function will affect the batch in the given order.
- **max_shard_size** (`str`, *optional*, defaults to `"10GB"`): Max model shard size to be used specified in a string.
- **safe_serialization** (`bool`, *optional*, defaults to `False`): Whetherto save model using safe tensors or the traditional PyTorch way. If `True`, some tensors will be lost.
- **compile** (`bool`, *optional*, defaults to `False`): Whether to call `torch.compile` on model (and teacher, if implemented).
- **train_loss_metric_name** (`str`, *optional*, defaults to `"train_loss"`): Metric name for train loss in logs.
- **val_loss_metric_name** (`str`, *optional*, defaults to `"val_loss"`): Metric name for validation loss in logs.
- **dataloader_pin_memory** (`bool`, *optional*, defaults to `True`): Enables pin memory option in DataLoader.
- **dataloader_num_workers** (`int`, *optional*, defaults to `0`): Number of processes for DataLoader.
- **report_loss_after_eval** (`bool`, *optional*, defaults to `True`): Whether to report average validation loss after evaluation. If set to `False`, loss will be reported by every batch.
- **handlers** (`Any` or `list`, *optional*, defaults to `None`): Handler or list of handlers to catch errors and make a safe checkpoint.
- **eval_when_finish** (`bool`, *optional*, defaults to `True`): At the end of training, evaluate model on validation dataset (if available). This option is only valid when `evaluate_every_n_steps` is not `None`.
- **eval_when_start** (`bool`, *optional*, defaults to `False`): Start training with evaluation (if available).
- **verbose** (`bool`, *optional*, defaults to `True`): (NOTE: This is a preliminary feature, and it may not disable prints in some accelerate backends). Enable or disable prints from Accelerate backend (e.g. training initialization, specific checkpoints, warnings, etc). You might want to enable this when debugging.
- **monitor** (`Monitor` or `dict`, *optional*, defaults to `None`): Monitor arguments to keep track of variables during training. If not specified, 'train_loss' and 'validation_loss' will be set to `True` by default. NOTE: Learning rate, GPU and CPU monitoring will only be reported during training, not evaluation. Also, GPU and CPU monitoring will only be reported on main process (index 0).
- **kwargs**  (`Any`, *optional*): Extra arguments for specific `init` function in Tracker, e.g. `run_name`, `tags`, etc.

## Train
To train a model, we must have initialized our Trainer. Then, we can call **fit** function. It requires the following arguments:
- **module** (AcceleratorModule): AcceleratorModule containing the training logic.
- **train_dataset** (`Dataset`, *optional*, defaults to `None`): Dataset class from PyTorch containing the train dataset logic.
- **val_dataset** (`Dataset`, *optional*, defaults to `None`): Dataset class from PyTorch containing the validation dataset logic.
- **kwargs** (`Any`, *optional*): Keyword arguments for `from_pretrained` function for model initialization (when using **transformers** models).

**train_dataset** and **val_dataset** are `None` by default because you can define the logic to get the DataLoader using **get_train_dataloader** and **get_validation_dataloader** in the AcceleratorModule subclass. If those are not defined, then the DataLoader will be created using an standard method.

**NOTE**: Only **train_dataset** or an implementation of **get_train_dataloader** is mandatory.
