import numpy as np
import pandas as pd
import sklearn.metrics


def confusion_score(y_true, y_pred, labels=[0, 1], normalize="true", **kwargs):
    cm = sklearn.metrics.confusion_matrix(
        y_true, y_pred, labels=labels, normalize=normalize, **kwargs
    )
    cm = pd.DataFrame(cm).set_index(
        pd.MultiIndex.from_tuples(list(zip(["True"] * len(labels), labels)))
    )
    cm.columns = pd.MultiIndex.from_tuples(
        list(zip(["Predicted"] * len(labels), labels))
    )
    return cm


def normalize_confusion_matrix(cm):
    for idx, row in cm.iterrows():
        cm.loc[idx] /= np.sum(row)


def mase(true, pred):
    if len(true) == 1:
        raise Exception("Length of sequence has to be at least 2")
    if type(true) is pd.Series:
        true = true.values
    if type(pred) is pd.Series:
        pred = pred.values
    diffs = np.abs(true[1:] - true[:-1])
    errors = np.abs(true - pred)
    return np.mean(errors) / np.mean(diffs)


def rmse(true, pred):
    return np.sqrt(sklearn.metrics.mean_squared_error(true, pred))


def mspe(true, pred):
    if type(true) is pd.Series:
        true = true.values
    if type(pred) is pd.Series:
        pred = pred.values
    errors = true - pred
    return np.mean(np.power((errors / true), 2))


def rmspe(true, pred):
    return np.sqrt(mspe(true, pred))


def auc(true, proba):
    if type(true) is pd.Series:
        true = true.values
    if type(proba) is pd.Series:
        proba = proba.values
    if type(proba) is pd.DataFrame:
        proba = proba.values
    if len(proba.shape) == 2 and proba.shape[1] == 2:
        proba = proba[:, 1]
    return sklearn.metrics.roc_auc_score(true, proba)
