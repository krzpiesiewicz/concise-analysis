import numpy as np
import pandas as pd

from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    roc_auc_score,
    r2_score,
    median_absolute_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    confusion_matrix,
)

from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold


def auc_score(clf, X, y):
    y_pred = clf.predict_proba(X)[:, 1]
    return roc_auc_score(y, y_pred)


def get_scores(
        model,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        X_test=None,
        y_test=None,
        train_pred_collector=None,
        val_pred_collector=None,
        test_pred_collector=None,
        X_challenge=None,
        challenge_pred_collector=None,
        get_confusion_matrix=False,
        classif=True,
        mape_score=True
):
    scores = dict()
    sets = [(X_train, y_train, "train", train_pred_collector)]
    if X_val is not None:
        sets.append((X_val, y_val, "val", val_pred_collector))
    if X_test is not None:
        sets.append((X_test, y_test, "test", test_pred_collector))
    for X, y, set_name, pred_collector in sets:
        y_pred = model.predict(X)
        if pred_collector is not None:
            pred_collector.add_preds(y.index, y_pred)
        if classif:
            acc = accuracy_score(y, y_pred)
            bacc = balanced_accuracy_score(y, y_pred)
            auc = auc_score(model, X, y)
            scores[set_name] = {"acc": acc, "bacc": bacc, "auc": auc}
            if get_confusion_matrix:
                scores[set_name]["confusion"] = confusion_score(y, y_pred)
        else:
            r2 = r2_score(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            medae = median_absolute_error(y, y_pred)
            scores[set_name] = {"r2": r2, "mae": mae, "medae": medae}
            if mape_score:
                mape = mean_absolute_percentage_error(y, y_pred)
                scores[set_name]["mape"] = mape
    if X_challenge is not None:
        y_pred = model.predict(X_challenge)
        if challenge_pred_collector is not None:
            challenge_pred_collector.add_preds(X_challenge.index, y_pred)
    return scores


def get_cv_scores(
        model_constructor,
        hparams,
        X,
        y,
        X_test=None,
        y_test=None,
        folds=5,
        n_repeats=1,
        random_state=87238232,
        train_pred_collector=None,
        val_pred_collector=None,
        test_pred_collector=None,
        X_challenge=None,
        challenge_pred_collector=None,
        classif=True,
        stratify=True,
        list_to_save_models=None,
        **kwargs
):
    mean_scores = dict()
    k_fold_contructor = (
        RepeatedStratifiedKFold if classif and stratify else RepeatedKFold
    )
    iterator = k_fold_contructor(
        n_splits=folds, random_state=random_state, n_repeats=n_repeats
    ).split(X, y)
    for train_idx, val_idx in iterator:
        train_idx = X.index[train_idx]
        val_idx = X.index[val_idx]
        X_train = X.loc[train_idx]
        y_train = y.loc[train_idx]
        X_val = X.loc[val_idx]
        y_val = y.loc[val_idx]
        model = model_constructor(**hparams)
        model.fit(X_train, y_train)
        if list_to_save_models is not None:
            list_to_save_models.append(model)
        scores = get_scores(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            train_pred_collector=train_pred_collector,
            val_pred_collector=val_pred_collector,
            test_pred_collector=test_pred_collector,
            X_challenge=X_challenge,
            challenge_pred_collector=challenge_pred_collector,
            classif=classif,
            **kwargs
        )
        for set_name, score_dct in scores.items():
            if not set_name in mean_scores:
                mean_scores[set_name] = dict()
            for metric, val in score_dct.items():
                if metric not in mean_scores[set_name]:
                    mean_scores[set_name][metric] = []
                mean_scores[set_name][metric].append(val)
    for set_name, score_dct in mean_scores.items():
        for metric in score_dct.keys():
            values = score_dct[metric]
            score_dct[metric] = dict()
            score_dct[metric]["std"] = np.std(values)
            score_dct[metric]["mean"] = np.mean(values)
    return mean_scores


def print_scores(scores, baseline_scores=None, return_str=False,
                 show_stds=False, newlines=2):
    str_buffer = ""
    j = 0
    for set_name, set_scores in scores.items():
        j += 1
        baseline_set_scores = None
        if baseline_scores is not None and set_name in baseline_scores:
            baseline_set_scores = baseline_scores[set_name]

        space = " " * (6 - len(set_name))
        i = 0
        for score_name, score in set_scores.items():
            i += 1
            score_mean = score["mean"] if type(score) is dict else score
            baseline_score = None
            if baseline_set_scores is not None and score_name in baseline_set_scores:
                baseline_score = baseline_set_scores[score_name]
                baseline_score_mean = baseline_score["mean"] if type(
                    baseline_score) is dict else baseline_score
            str_buffer += f"{set_name}_{score_name}:{space} {score_mean:.4f}"
            if baseline_score is not None:
                diff = score_mean - baseline_score_mean
                if abs(diff) <= 1e-4:
                    diff = 0
                change = "=" if abs(diff) <= 1e-4 else (
                    "+" if diff > 0 else "-")
                str_buffer += f" ({change}{abs(diff):.4f})"
            if i < len(set_scores):
                str_buffer += " " * 7
        if show_stds:
            str_buffer += "\n"
            i = 0
            for score_name, score in set_scores.items():
                i += 1
                score_std = score["std"] if type(score) is dict else score
                baseline_score_std = None
                baseline_score = None
                if baseline_set_scores is not None and score_name in baseline_set_scores:
                    baseline_score = baseline_set_scores[score_name]
                    baseline_score_std = baseline_score["std"] if type(
                        baseline_score) is dict else None
                space_std = " " * (6 + 1 + len(score_name) + 1 - len("std:"))
                str_buffer += f"{' ' * len('std:')}{space_std}±{score_std:.4f}"
                if baseline_score_std is not None:
                    diff = score_std - baseline_score_std
                    if abs(diff) <= 1e-4:
                        diff = 0
                    change = "=" if abs(diff) <= 1e-4 else (
                        "+" if diff > 0 else "-")
                    str_buffer += f" ({change}{abs(diff):.4f})"
                elif baseline_score is not None:
                    str_buffer += " " * (5 + 5)
                if i < len(set_scores):
                    str_buffer += " " * 7
        if j < len(scores):
            str_buffer += "\n\n"
    str_buffer += "\n" * newlines
    if return_str:
        return str_buffer
    else:
        print(str_buffer)


def confusion_score(y_true, y_pred, labels=[0, 1], normalize="true", **kwargs):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize,
                          **kwargs)
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
