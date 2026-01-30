import numpy as np
import pandas as pd

def evaluate(results_storage, metrics, labels, score, **args):
    if "accomplish_UCR" in metrics:
        result = {}
        sorted_indexs = np.argsort(-score)        # DESC
        # find the first overlap index
        topk = 0
        for i, (index) in enumerate(sorted_indexs):
            if labels[index] == 1:
                topk = i + 1
                break
            
        result['topk'] = topk
        result['total_len'] = len(score)
        aplha_quantile = topk/len(score)
        result['aplha_quantile'] = aplha_quantile
        result['3_alpha'] = aplha_quantile < 0.03
        result['10_alpha'] = aplha_quantile < 0.1
        results_storage['accomplish_UCR'] = pd.DataFrame([result])