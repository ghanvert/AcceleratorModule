import operator
from dataclasses import dataclass

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
