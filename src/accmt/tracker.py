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

from abc import ABC, abstractmethod
from typing import Any, Literal, Optional, Union

from accelerate.utils import LoggerType

from .utility import ASYNC, ASYNC_TRAIN_GROUP, MASTER_PROCESS


_logger_type_map = {
    "mlflow": LoggerType.MLFLOW,
    # "tensorboard": LoggerType.TENSORBOARD,
    # "wandb": LoggerType.WANDB,
    # "cometml": LoggerType.COMETML,
    # "aim": LoggerType.AIM,
    # "clearml": LoggerType.CLEARML,
    # "dvclive": LoggerType.DVCLIVE,
}


class BaseTracker(ABC):
    def __init__(self, tracker: Optional[Union[LoggerType, str]] = None):
        if tracker is None:
            raise ValueError("'tracker' cannot be `None`.")
        self.logger_type = tracker if not isinstance(tracker, str) else _logger_type_map[tracker]
        self.name: str = self.logger_type.value

    def get_init_kwargs(self, **kwargs) -> dict:
        return {self.name: {**kwargs}}

    @property
    @abstractmethod
    def run_id(self) -> str:
        pass

    @abstractmethod
    def set_tracking_uri(self, uri: str):
        pass

    @abstractmethod
    def log_artifact(self, path: str):
        pass

    @abstractmethod
    def log_artifacts(self, path: str):
        pass

    @abstractmethod
    def log(self, metrics: dict[str, Any], step: int, run_id: Optional[str] = None):
        pass

    @abstractmethod
    def end(self, status: Literal["FINISHED", "FAILED", "KILLED"] = "FINISHED"):
        pass


class MLFlow(BaseTracker):
    def __init__(self):
        super().__init__("mlflow")
        import mlflow

        self.module = mlflow

    @property
    def run_id(self) -> str:
        return self.module.active_run().info.run_id

    def set_tracking_uri(self, uri):
        if MASTER_PROCESS:
            self.module.set_tracking_uri(uri)

    def log_artifact(self, path):
        if ASYNC and not ASYNC_TRAIN_GROUP:
            return

        if MASTER_PROCESS:
            self.module.log_artifact(path)

    def log_artifacts(self, path):
        if ASYNC and not ASYNC_TRAIN_GROUP:
            return

        if MASTER_PROCESS:
            self.module.log_artifacts(path)

    def log(self, metrics, step, run_id=None):
        if MASTER_PROCESS:
            self.module.log_metrics(metrics, step=step, run_id=run_id)

    def end(self, status: Literal["FINISHED", "FAILED", "KILLED"] = "FINISHED"):
        if ASYNC and not ASYNC_TRAIN_GROUP:
            return

        try:
            from mlflow.entities import RunStatus

            self.module.end_run(status=RunStatus.to_string(getattr(RunStatus, status, RunStatus.FINISHED)))
        except Exception:
            pass  # ignore errors for this tracker


_tracker_map = {
    "mlflow": MLFlow,
}
