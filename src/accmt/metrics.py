import operator
from dataclasses import dataclass

@dataclass
class MetricComparator:
    accuracy = operator.gt
    bertscore = operator.gt
    bleu = operator.gt
    bleurt = operator.gt
    brier_score = operator.lt
    cer = operator.lt
    character = operator.gt
    charcut_mt = operator.lt
    chrf = operator.gt
    f1 = operator.gt
    glue = operator.gt
    precision = operator.gt
    r_squared = operator.gt
    recall = operator.gt
    mse = operator.lt
    mean_iou = operator.gt
