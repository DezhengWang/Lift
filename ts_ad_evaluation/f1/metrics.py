import numpy as np
import sklearn
import sklearn.preprocessing

def adjustment(gt, pred):
    adjusted_pred = np.array(pred)
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and adjusted_pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if adjusted_pred[j] == 0:
                        adjusted_pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if adjusted_pred[j] == 0:
                        adjusted_pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            adjusted_pred[i] = 1
    return adjusted_pred


class metricor:
    def __init__(self, bias = 'flat'):
        self.bias = bias
        self.eps = 1e-15

    def w(self, AnomalyRange, p):
        MyValue = 0
        MaxValue = 0
        start = AnomalyRange[0]
        AnomalyLength = AnomalyRange[1] - AnomalyRange[0] + 1
        for i in range(start, start +AnomalyLength):
            bi = self.b(i, AnomalyLength)
            MaxValue +=  bi
            if i in p:
                MyValue += bi
        return MyValue/MaxValue
    
    def Cardinality_factor(self, Anomolyrange, Prange):
        score = 0
        start = Anomolyrange[0]
        end = Anomolyrange[1]
        for i in Prange:
            if i[0] >= start and i[0] <= end:
                score +=1
            elif start >= i[0] and start <= i[1]:
                score += 1
            elif end >= i[0] and end <= i[1]:
                score += 1
            elif start >= i[0] and end <= i[1]:
                score += 1
        if score == 0:
            return 0
        else:
            return 1/score
    
    def b(self, i, length):
        bias = self.bias
        if bias == 'flat':
            return 1
        elif bias == 'front-end bias':
            return length - i + 1
        elif bias == 'back-end bias':
            return i
        else:
            if i <= length/2:
                return i
            else:
                return length - i + 1
    
    def metric_RF1(self, label, preds):
        Rrecall, ExistenceReward, OverlapReward = self.range_recall_new(label, preds, alpha=0.2)
        Rprecision = self.range_recall_new(preds, label, 0)[0]
        if Rprecision + Rrecall==0:
            RF1=0
        else:
            RF1 = 2 * Rrecall * Rprecision / (Rprecision + Rrecall)
        return Rprecision, Rrecall, RF1
    
    def range_recall_new(self, labels, preds, alpha):
        p = np.where(preds == 1)[0]    # positions of predicted label==1
        range_pred = self.range_convers_new(preds)
        range_label = self.range_convers_new(labels)

        Nr = len(range_label)    # total # of real anomaly segments

        ExistenceReward = self.existence_reward(range_label, preds)


        OverlapReward = 0
        for i in range_label:
            OverlapReward += self.w(i, p) * self.Cardinality_factor(i, range_pred)


        score = alpha * ExistenceReward + (1-alpha) * OverlapReward
        if Nr != 0:
            return score/Nr, ExistenceReward/Nr, OverlapReward/Nr
        else:
            return 0,0,0

    def range_convers_new(self, label):
        '''
        input: arrays of binary values
        output: list of ordered pair [[a0,b0], [a1,b1]... ] of the inputs
        '''
        anomaly_starts = np.where(np.diff(label) == 1)[0] + 1
        anomaly_ends, = np.where(np.diff(label) == -1)
        if len(anomaly_ends):
            if not len(anomaly_starts) or anomaly_ends[0] < anomaly_starts[0]:
                # we started with an anomaly, so the start of the first anomaly is the start of the labels
                anomaly_starts = np.concatenate([[0], anomaly_starts])
        if len(anomaly_starts):
            if not len(anomaly_ends) or anomaly_ends[-1] < anomaly_starts[-1]:
                # we ended on an anomaly, so the end of the last anomaly is the end of the labels
                anomaly_ends = np.concatenate([anomaly_ends, [len(label) - 1]])
        return list(zip(anomaly_starts, anomaly_ends))
    
    def existence_reward(self, labels, preds):
        '''
        labels: list of ordered pair
        preds predicted data
        '''

        score = 0
        for i in labels:
            if preds[i[0]:i[1]+1].any():
                score += 1
        return score
    
    def _get_events(self, y_test, outlier=1, normal=0):
        events = dict()
        label_prev = normal
        event = 0  # corresponds to no event
        event_start = 0
        for tim, label in enumerate(y_test):
            if label == outlier:
                if label_prev == normal:
                    event += 1
                    event_start = tim
            else:
                if label_prev == outlier:
                    event_end = tim - 1
                    events[event] = (event_start, event_end)
            label_prev = label

        if label_prev == outlier:
            event_end = tim - 1
            events[event] = (event_start, event_end)
        return events

    def metric_EventF1PA(self, label, preds):
        from sklearn.metrics import precision_score
        true_events = self._get_events(label)

        tp = np.sum([preds[start:end + 1].any() for start, end in true_events.values()])
        fn = len(true_events) - tp
        rec_e = tp/(tp + fn)
        prec_t = precision_score(label, preds)
        EventF1PA1 = 2 * rec_e * prec_t / (rec_e + prec_t + self.eps)

        return prec_t, rec_e, EventF1PA1


import pandas as pd
def evaluate(results_storage, metrics, labels, score, **args):
    if "best_f1" in metrics:
        result = {}
        Ps, Rs, thres = sklearn.metrics.precision_recall_curve(labels, score)
        F1s = (2 * Ps * Rs) / (Ps + Rs)
        best_F1_index = np.argmax(F1s[np.isfinite(F1s)])
        best_thre = thres[best_F1_index]
        pred = (score > best_thre).astype(int)
        best_acc = sklearn.metrics.accuracy_score(labels, pred)
        result['thre_best'] = best_thre
        result['ACC_best'] = best_acc
        result['P_best'] = Ps[best_F1_index] 
        result['R_best'] = Rs[best_F1_index] 
        result['F1_best'] = F1s[best_F1_index]
        results_storage['best_f1'] = pd.DataFrame([result])
    if "f1_raw" in metrics:
        results = []
        for thre in args['f1_raw']:
            result = {}
            pred = (score > thre).astype(int)
            accuracy = sklearn.metrics.accuracy_score(labels, pred)
            P, R, F1, _ = sklearn.metrics.precision_recall_fscore_support(labels, pred, average="binary")
            result['thre_raw'] = thre
            result['ACC_raw'] = accuracy
            result['P_raw'] = P 
            result['R_raw'] = R 
            result['F1_raw'] = F1
            results.append(pd.DataFrame([result]))
        results_storage['f1_raw'] = pd.concat(results, axis=0).reset_index(drop=True)
    if "f1_pa" in metrics:
        results = []
        for thre in args['f1_pa']:
            result = {}
            pred = (score > thre).astype(int)
            adjusted_pred = adjustment(labels, pred)
            accuracy = sklearn.metrics.accuracy_score(labels, adjusted_pred)
            P, R, F1, _ = sklearn.metrics.precision_recall_fscore_support(labels, adjusted_pred, average="binary")
            result['thre_PA'] = thre
            result['ACC_PA'] = accuracy
            result['P_PA'] = P 
            result['R_PA'] = R 
            result['F1_PA'] = F1
            results.append(pd.DataFrame([result]))
        results_storage['f1_pa'] = pd.concat(results, axis=0).reset_index(drop=True)
    if "f1_r" in metrics:
        results = []
        for thre in args['f1_r']:
            result = {}
            pred = (score > thre).astype(int)
            accuracy = sklearn.metrics.accuracy_score(labels, pred)
            PR, RR, F1R= metricor().metric_RF1(labels, pred)
            result['thre_r'] = thre
            result['ACC_r'] = accuracy
            result['P_r'] = PR
            result['R_r'] = RR 
            result['F1_r'] = F1R
            results.append(pd.DataFrame([result]))
        results_storage['f1_r'] = pd.concat(results, axis=0).reset_index(drop=True)
    if "f1_event" in metrics:
        results = []
        for thre in args['f1_event']:
            result = {}
            pred = (score > thre).astype(int)
            accuracy = sklearn.metrics.accuracy_score(labels, pred)
            PE, RE, F1E= metricor().metric_EventF1PA(labels, pred)
            result['thre_event'] = thre
            result['ACC_event'] = accuracy
            result['P_event'] = PE
            result['R_event'] = RE 
            result['F1_event'] = F1E
            results.append(pd.DataFrame([result]))
        results_storage['f1_event'] = pd.concat(results, axis=0).reset_index(drop=True)
