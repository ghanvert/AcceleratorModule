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

import evaluate

from accmt.metrics import Metric


class Accuracy(Metric):
    def __init__(self, *args, **kwargs):
        # we are extending the 'Metric' class to initialize the 'evaluate' module, so we need
        # to call the parent class constructor.
        super().__init__(*args, **kwargs)
        self.accuracy_module = evaluate.load("accuracy")

    def compute(self, predictions, references):
        # here, predictions and references will be a 2D tensor, where the first dimension is the batch size.

        # you can compute metrics as you want, in this case, we use the accuracy metric from 'evaluate' library.
        score = self.accuracy_module.compute(predictions=predictions, references=references)["accuracy"]

        return {"accuracy": score}
