import operator
import torch
from dataclasses import dataclass
from typing import Optional, Iterable, Callable
from evaluate import load, EvaluationModule

@dataclass
class MetricComparator:
    accuracy = operator.ge
    bertscore = operator.ge
    bleu = operator.ge
    bleurt = operator.ge
    brier_score = operator.le
    cer = operator.le
    character = operator.ge
    charcut_mt = operator.le
    chrf = operator.ge
    f1 = operator.ge
    glue = operator.ge
    precision = operator.ge
    r_squared = operator.ge
    recall = operator.ge
    mse = operator.le
    mean_iou = operator.ge
    wer = operator.le

class Metric:
    def __init__(self):
        self.module: EvaluationModule = None
        self.predictions = []
        self.references = []

    def compute(self,
                *,
                predictions: Optional[Iterable] = None,
                references: Optional[Iterable] = None,
                custom_function: Optional[Callable[[Iterable, Iterable], dict]] = None
    ) -> dict:
        """
        Compute metric with predictions and references.
        """
        metric_str_error = "If not loading evaluation module from HuggingFace's Evaluate, you must give a 'custom_function'."
        assert (self.module is not None and custom_function is None) or (self.module is None and custom_function is not None), metric_str_error

        if predictions is not None and references is not None:
            self.predictions = predictions
            self.references = references

        assert len(self.predictions) == len(self.references), "Predictions and references must be of the same length."

        if self.module is None: # custom metrics
            output = custom_function(self.predictions, self.references)
        else: # Evaluate's library
            output = self.module.compute()

        self.clear()

        return output

    @classmethod
    def from_hf(cls, module: str, **kwargs):
        evaluate_module = load(module, **kwargs)
        metric = Metric()
        metric.module = evaluate_module

        return metric
        
    def clear(self):
        self.predictions = []
        self.references = []

    def add_batch(self, *, predictions = None, references = None, **kwargs):
        if self.module is None: # custom metrics
            self.predictions.append(predictions)
            self.references.append(references)
        else: # Evaluate's librar
            self.module.add_batch(predictions=predictions, references=references, **kwargs)

    def cat(self, dim: int = 0):
        self.predictions = torch.cat(self.predictions, dim=dim)
        self.references = torch.cat(self.references, dim=dim)
