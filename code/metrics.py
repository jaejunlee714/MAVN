import numpy as np
from sklearn.metrics import average_precision_score, f1_score, mean_absolute_error, roc_auc_score, accuracy_score

def AP(pred, y):
    aps = []
    for class_idx in range(len(y[0])):
        aps.append(average_precision_score(y_true = y[:,class_idx], y_score = pred[:, class_idx]))
    return sum(aps)/len(aps)

def Macro_F1(pred, y):
    return f1_score(y_true = y, y_pred = pred, average = 'macro')

def MAE(pred, y):
    return mean_absolute_error(y_true = y, y_pred = pred)

def Accuracy(pred, y):
    return accuracy_score(y_true = y, y_pred = pred)

def AUCROC(pred, y):
    return roc_auc_score(y_true = y, y_score=pred)