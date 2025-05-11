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

from src.accmt import set_seed
from src.accmt.hp_search import HyperParameterSearch

from .dummy_dataset import DummyDataset
from .dummy_metrics import Accuracy, DictMetrics
from .train import DummyModule


if __name__ == "__main__":
    set_seed(42)

    def get_module():
        return DummyModule()

    train_dataset = DummyDataset()
    val_dataset = DummyDataset()

    metrics = [Accuracy("accuracy"), DictMetrics("test_dict")]

    def get_best_metric(additional_metrics):
        return additional_metrics["0"]["accuracy"]

    hp_search = HyperParameterSearch(get_module, train_dataset, val_dataset, metrics)
    hp_search.set_parameters(
        train_batch_size=16,
        learning_rate=[1e-6, 1e-4],
    )
    hp_search.optimize(get_best_metric, direction="maximize", n_trials=3)
