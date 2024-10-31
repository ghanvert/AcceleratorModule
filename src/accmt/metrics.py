import torch
from typing_extensions import Literal, Optional, override

_available_comparators = ['<', '<=', '>', '>=', '==']

class Metric:
    def __init__(self, name: str, comparator: Literal['<', '<=', '>', '>=', '=='] = '>', main_metric: Optional[str] = None):
        """
        Set a module to compute metrics.

        Args:
            name (`str`):
                Metric's module name.
            comparator (`str`, *optional*, defaults to `>`):
                Metric comparator to determine if current main metric value is the best calculated. Available 
                options are: '<', '<=', '>', '>=' and '=='. For example, if set to '>', the comparation will be 
                a > b, 'a' being current value and 'b' being previous value.
            main_metric (`str`, *optional*, defaults to `None`):
                Determine which is the main metric key in your compute output. By default, main metric key will be 
                equal to the 'name' parameter.
        """
        self.name = name
        assert comparator in _available_comparators, f"Available options for comparator are: {_available_comparators}"
        self.comparator = comparator
        self.main_metric = main_metric if main_metric is not None else name

        self.predictions = []
        self.references = []

    @override
    def compute(self, predictions: torch.Tensor, references: torch.Tensor) -> dict:
        """
        Compute metrics with given predictions and references. This function returns a dictionary 
        containing the main metric value and others.
        
        Example:
            ```
            def compute(self, predictions, references):
                # logic of how to calculate metrics here...

                return {
                    "accuracy": 0.85, # <-- this one is the main value
                    "f1": 0.89
                }
            ```

        NOTE: In the previous example, the main metric is 'accuracy', and its value is gonna be used along with 
        'comparator' to compare if the metric is the best or not. By default, main metric is set to the name of 
        the metric itself. You can change this behaviour with 'main_metric' on class initialization.
        """

    def _compute(self) -> dict:
        assert len(self.predictions) == len(self.references), "Predictions and references must be of the same length."

        self._cat()
        output = self.compute(self.predictions, self.references)
        self.clear()

        return output
        
    def clear(self):
        self.predictions = []
        self.references = []

    def add_batch(self, *, predictions = None, references = None):
        self.predictions.append(predictions)
        self.references.append(references)

    def _cat(self):
        self.predictions = torch.cat(self.predictions)
        self.references = torch.cat(self.references)
