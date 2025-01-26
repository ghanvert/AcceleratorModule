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

from dataclasses import dataclass

from accelerate.utils import LoggerType


@dataclass
class TensorBoard:
    tracker = LoggerType.TENSORBOARD

    def init(**kwargs):
        return {"tensorboard": {**kwargs}}


@dataclass
class WandB:
    tracker = LoggerType.WANDB

    def init(**kwargs):
        return {"wandb": {**kwargs}}


@dataclass
class CometML:
    tracker = LoggerType.COMETML

    def init(**kwargs):
        return {"comet_ml": {**kwargs}}


@dataclass
class Aim:
    tracker = LoggerType.AIM

    def init(**kwargs):
        return {"aim": {**kwargs}}


@dataclass
class MLFlow:
    tracker = LoggerType.MLFLOW

    def init(**kwargs):
        return {"mlflow": {**kwargs}}


@dataclass
class ClearML:
    tracker = LoggerType.CLEARML

    def init(**kwargs):
        return {"clearml": {**kwargs}}


@dataclass
class DVCLive:
    tracker = LoggerType.DVCLIVE

    def init(**kwargs):
        return {"dvclive": {**kwargs}}
