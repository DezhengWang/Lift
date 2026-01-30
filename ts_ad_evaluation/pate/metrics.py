from .PATE_metric import PATE

import pandas as pd
def evaluate(results_storage, metrics, labels, score, **args):
    if "pate" in metrics:
        result = {}
        pate = PATE(labels, score, binary_scores = False, n_jobs=-1, num_splits_MaxBuffer=1, include_zero=False)
        result['pate'] = pate
        results_storage['pate'] = pd.DataFrame([result])
        