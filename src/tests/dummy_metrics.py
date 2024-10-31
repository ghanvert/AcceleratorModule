from src.accmt.metrics import Metric

class Accuracy(Metric):
    def compute(self, predictions, references):
        print(predictions.shape)
        print(references.shape)
        return {
            "accuracy": 0.85,
            "test_metric": 0.5
        }
