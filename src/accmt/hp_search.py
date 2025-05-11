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

from collections.abc import Sequence
from typing import Callable, Literal, Optional, Union

import optuna
import torch.distributed as dist
from torch.utils.data import Dataset

from accmt.metrics import Metric
from accmt.utility import MASTER_PROCESS, WORLD_SIZE

from .hyperparameters import HyperParameters, Optimizer, Scheduler
from .trainer import Trainer


class HPSTrainer(Trainer):
    def __init__(self, hps: HyperParameters, metrics: list[Metric] = None, **kwargs):
        super().__init__(
            hps_config=hps,
            metrics=metrics,
            model_path="_dummy_model_hp_search",
            enable_checkpointing=False,
            disable_model_saving=True,
            destroy_after_training=False,
            **kwargs,
        )


class HyperParameterSearch:
    def __init__(
        self,
        get_module_fn: Callable,
        train_dataset: Dataset,
        val_dataset: Union[Dataset, list[Dataset], dict[str, Dataset]],
        metrics: Optional[list[Metric]] = None,
        **trainer_kwargs,
    ):
        """
        Initialize the hyperparameter search.

        Args:
            get_module_fn (`Callable`):
                A function that returns an `AcceleratorModule` (basically, the model initialization). It does
                not take any arguments.
            train_dataset (`Dataset`):
                The training dataset.
            val_dataset (`Dataset` or `list[Dataset]` or `dict[str, Dataset]`):
                The validation dataset(s).
            metrics (`list`, *optional*, defaults to `None`):
                The metrics modules to evaluate. If not provided, the default metric will be used (`valid_loss`).
            **trainer_kwargs:
                Additional keyword arguments to pass to the `Trainer` constructor.
        """
        self.params = {}  # must be set using `set_parameters`

        self.get_module_fn = get_module_fn
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.metrics = metrics
        self.trainer_kwargs = trainer_kwargs

        self.best_metric_fn: Callable = None  # must be set using `optimize`

    def get_hps(self, trial: optuna.Trial) -> HyperParameters:
        """
        Get the hyperparameter space. Must set the parameters first using `set_parameters`.

        Args:
            trial (`optuna.Trial`): The trial object.

        Returns:
            `HyperParameters`: The hyperparameter space for the Adam or AdamW optimizer.
        """
        # Check if at least one parameter is a sequence
        if not any(
            isinstance(self.params[p], Sequence)
            for p in [
                "train_batch_size",
                "epochs",
                "max_steps",
                "learning_rate",
                "beta1",
                "beta2",
                "eps",
                "weight_decay",
                "warmup_ratio",
            ]
        ) and not isinstance(self.params["scheduler"], dict):
            raise ValueError("At least one of the arguments must be a range (`list`).")

        if self.params["optimizer"] not in {"AdamW", "Adam"}:
            raise ValueError("Optimizer must be either 'AdamW' or 'Adam'.")

        # Process parameters that might be ranges
        param_names = [
            "train_batch_size",
            "epochs",
            "max_steps",
            "learning_rate",
            "beta1",
            "beta2",
            "eps",
            "weight_decay",
            "warmup_ratio",
        ]

        params = self.params.copy()
        for param in param_names:
            if isinstance(params[param], Sequence):
                if isinstance(params[param], tuple) or param == "train_batch_size":
                    params[param] = trial.suggest_categorical(param, params[param])
                elif param in ["epochs", "max_steps"]:
                    params[param] = trial.suggest_int(param, *params[param])
                else:
                    params[param] = trial.suggest_float(param, *params[param])

        # Handle scheduler if it's a dictionary of options
        if self.params["scheduler"] is not None:
            if isinstance(self.params["scheduler"], dict):
                _scheduler = trial.suggest_categorical("scheduler", list(self.params["scheduler"].keys()))
                params["scheduler"] = self.params["scheduler"][_scheduler]
            elif isinstance(self.params["scheduler"], Sequence):
                raise ValueError("Scheduler must be a dictionary of options or fixed scheduler.")

        # Set eval batch size if not provided
        if self.params["eval_batch_size"] is None:
            params["eval_batch_size"] = params["train_batch_size"]

        return HyperParameters(
            epochs=params["epochs"],
            max_steps=params["max_steps"],
            batch_size=[params["train_batch_size"], params["eval_batch_size"]],
            optimizer=getattr(Optimizer, params["optimizer"]),
            optim_kwargs={
                "lr": params["learning_rate"],
                "betas": (params["beta1"], params["beta2"]),
                "eps": params["eps"],
                "weight_decay": params["weight_decay"],
            },
            scheduler=params["scheduler"],
            scheduler_kwargs={"warmup_ratio": params["warmup_ratio"]},
            step_scheduler_per_epoch=params["step_scheduler_per_epoch"],
        )

    def set_parameters(
        self,
        *,
        train_batch_size: Union[int, list[int]] = 1,
        epochs: Union[int, list[int]] = 1,
        max_steps: Union[int, list[int]] = None,
        learning_rate: Union[float, list[float]] = [1e-3, 1e-7],
        beta1: Union[float, list[float]] = 0.9,
        beta2: Union[float, list[float]] = 0.999,
        eps: Union[float, list[float]] = 1e-08,
        weight_decay: Union[float, list[float]] = 0.01,
        scheduler: Optional[Union[Scheduler, dict]] = None,
        warmup_ratio: Union[float, list[float]] = 0.1,
        step_scheduler_per_epoch: bool = False,
        optimizer: Literal["AdamW", "Adam"] = "AdamW",
        eval_batch_size: Optional[int] = None,
    ):
        """
        Set the parameters for the hyperparameter search. Fixed values are represented as a single value. A range
        of parameters is represented as a list of values. A tuple of values means discrete values.

        Args:
            train_batch_size (`int` or `list`, *optional*, defaults to `1`):
                The batch size for the training set. This is a categorical argument, so Optuna will sample
                from the list of options.
            epochs (`int` or `list`, *optional*, defaults to `1`):
                The number of epochs to train the model.
            max_steps (`int` or `list`, *optional*, defaults to `None`):
                The maximum number of steps to train the model. Drop-in replacement for `epochs`.
            learning_rate (`float` or `list`, *optional*, defaults to `[1e-3, 1e-7]`):
                The learning rate to use for the optimizer.
            beta1 (`float` or `list`, *optional*, defaults to `0.9`):
                The beta1 parameter for the Adam optimizer.
            beta2 (`float` or `list`, *optional*, defaults to `0.999`):
                The beta2 parameter for the Adam optimizer.
            eps (`float` or `list`, *optional*, defaults to `1e-08`):
                The epsilon parameter for the Adam optimizer.
            weight_decay (`float` or `list`, *optional*, defaults to `0.01`):
                The weight decay to use for the optimizer.
            scheduler (`dict` or `Scheduler`, *optional*, defaults to `None`):
                The learning rate scheduler to use for the optimizer. This is a dictionary containing the
                scheduler name as a key and the scheduler object as a value. Do not consider this argument as
                a range, but a list of options to choose from.
            warmup_ratio (`float` or `list`, *optional*, defaults to `0.1`):
                The warmup ratio to use for the learning rate scheduler. Only used if `scheduler` is not `None`
                and is a warmup scheduler.
            step_scheduler_per_epoch (`bool`, *optional*, defaults to `False`):
                Whether to step the scheduler per epoch or per step. This is a fixed value.
            optimizer (`Literal["AdamW", "Adam"]`, *optional*, defaults to `"AdamW"`):
                The optimizer to use for the training (not a range, just a fixed value).
            eval_batch_size (`int`, *optional*, defaults to `None`):
                The batch size for the evaluation set. If not provided, the training batch size will be used.
                NOTE: This is not a hyperparameter, so it should be a fixed value.
        """
        self.params = {
            "train_batch_size": train_batch_size,
            "epochs": epochs,
            "max_steps": max_steps,
            "learning_rate": learning_rate,
            "beta1": beta1,
            "beta2": beta2,
            "eps": eps,
            "weight_decay": weight_decay,
            "scheduler": scheduler,
            "warmup_ratio": warmup_ratio,
            "step_scheduler_per_epoch": step_scheduler_per_epoch,
            "optimizer": optimizer,
            "eval_batch_size": eval_batch_size,
        }

    def objective_fn(self, trial: optuna.Trial):
        if len(self.params) == 0:
            raise ValueError("No parameters set. Please set the parameters first using `set_parameters`.")

        module = self.get_module_fn()
        hps = self.get_hps(trial)
        trainer = HPSTrainer(hps, self.metrics, **self.trainer_kwargs)
        trainer.fit(module, self.train_dataset, self.val_dataset)

        score = self.best_metric_fn(trainer.state.additional_metrics)
        return score

    def _get_default_best_metric(self, additional_metrics: dict):
        return additional_metrics["0"]["valid_loss"]

    def optimize(
        self,
        best_metric_fn: Optional[Callable] = None,
        direction: Literal["maximize", "minimize"] = "minimize",
        n_trials: int = 10,
    ):
        """
        Optimize an objective function.

        Args:
            best_metric_fn (`Callable`, *optional*, defaults to `None`):
                A function that takes a dictionary of additional metrics and returns the best metric. This
                function receives a single argument, which is a dictionary of metrics. Must return a float.
            n_trials (`int`, *optional*, defaults to `10`):
                The number of trials to run.
        """
        direction = direction.lower()
        if direction not in {"maximize", "minimize"}:
            raise ValueError("Direction must be either 'maximize' or 'minimize'.")

        self.best_metric_fn = best_metric_fn if best_metric_fn is not None else self._get_default_best_metric

        if not MASTER_PROCESS:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(direction=direction, study_name="Hyper Parameter Search")
        study.optimize(self.objective_fn, n_trials=n_trials)

        # print best value and best params
        if MASTER_PROCESS:
            print("\n")
            print("=" * 100)
            print("Best results:")
            print(f"\tBest value: {study.best_value}")
            print("\tBest parameters:")
            for param, value in study.best_params.items():
                print(f"\t\t{param}: {value}")
            print("=" * 100)

        if WORLD_SIZE > 1:
            dist.destroy_process_group()
