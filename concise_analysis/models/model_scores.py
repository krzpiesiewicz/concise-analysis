import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold

from concise_analysis.metrics import get_scoring


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
        predict_proba=False,
        auc_score=True,
        mape_score=True,
        metrics="default",
        extra_metrics=None,
        dependent_metrics=None  # a list of functions which calculate extra
                    # metrics depending on basic ones.
                    # Signature:
                    # dict(metric_name, value) -> (new_name, new_value)
):
    if metrics == "default":
        if classif:
            metrics = ["acc", "bacc"]
            if auc_score:
                metrics += ["auc"]
            if get_confusion_matrix:
                metrics += ["confusion"]
        else:
            metrics = ["r2", "mae", "mse", "medae"]
            if mape_score:
                metrics += ["mape"]
    if extra_metrics is not None:
        metrics += extra_metrics
    scores = dict()
    sets = [(X_train, y_train, "train", train_pred_collector)]
    if X_val is not None:
        sets.append((X_val, y_val, "val", val_pred_collector))
    if X_test is not None:
        sets.append((X_test, y_test, "test", test_pred_collector))

    for X, y, set_name, pred_collector in sets:
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(
            X) if classif and predict_proba else None
        if pred_collector is not None:
            pred_collector.add_preds(y.index, y_pred, y_pred_proba)
        scores[set_name] = {}

        for metric in metrics:
            scoring = get_scoring(metric)
            metric_name = metric if type(metric) is str else scoring["name"]
            score_func = scoring["scoring"]
            if scoring["pred_type"] == "proba":
                if y_pred_proba is None:
                    y_pred_proba = model.predict_proba(X)
                score = score_func(y, y_pred_proba)
            else:
                score = score_func(y, y_pred)
            scores[set_name][metric_name] = score

        if dependent_metrics is not None:
            for get_depnt_score in dependent_metrics:
                metric_name, score = get_depnt_score(scores[set_name])
                scores[set_name][metric_name] = score

    if X_challenge is not None:
        y_pred = model.predict(X_challenge)
        y_pred_proba = model.predict_proba(
            X_challenge) if classif and predict_proba else None
        if challenge_pred_collector is not None:
            challenge_pred_collector.add_preds(X_challenge.index, y_pred,
                                               y_pred_proba)
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
        **kwargs,
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
            **kwargs,
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
            n = len(values)
            mean_values = sum(values) / n
            score_dct[metric]["mean"] = mean_values
            score_dct[metric]["std"] = (
                    sum([d ** 2 for d in
                         [val - mean_values for val in values]]) ** (0.5)
                    / n
            )
    return mean_scores


def print_scores(
        scores, baseline_scores=None, return_str=False, show_stds=False,
        newlines=2, space=7
):
    str_buffer = ""
    j = 0
    for set_name, set_scores in scores.items():
        j += 1
        baseline_set_scores = None
        if baseline_scores is not None and set_name in baseline_scores:
            baseline_set_scores = baseline_scores[set_name]

        setname_space = " " * (6 - len(set_name))
        i = 0
        for score_name, score in set_scores.items():
            score_mean = score["mean"] if type(score) is dict else score
            if np.isscalar(score_mean):
                i += 1
                baseline_score = None
                if baseline_set_scores is not None and score_name in baseline_set_scores:
                    baseline_score = baseline_set_scores[score_name]
                    baseline_score_mean = (
                        baseline_score["mean"]
                        if type(baseline_score) is dict
                        else baseline_score
                    )
                str_buffer += f"{set_name}_{score_name}:{setname_space} {score_mean:.4f}"
                if baseline_score is not None:
                    diff = score_mean - baseline_score_mean
                    if abs(diff) <= 1e-4:
                        diff = 0
                    change = "=" if abs(diff) <= 1e-4 else (
                        "+" if diff > 0 else "-")
                    str_buffer += f" ({change}{abs(diff):.4f})"
                if i < len(set_scores):
                    str_buffer += " " * space
        if show_stds:
            str_buffer += "\n"
            i = 0
            for score_name, score in set_scores.items():
                score_std = score["std"] if type(score) is dict else score
                if np.isscalar(score_std):
                    i += 1
                    baseline_score_std = None
                    baseline_score = None
                    if (
                            baseline_set_scores is not None
                            and score_name in baseline_set_scores
                    ):
                        baseline_score = baseline_set_scores[score_name]
                        baseline_score_std = (
                            baseline_score["std"] if type(
                                baseline_score) is dict else None
                        )
                    std_space = " " * (
                            6 + 1 + len(score_name) + 1 - len("std:"))
                    str_buffer += f"{' ' * len('std:')}{std_space}±{score_std:.4f}"
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
                        str_buffer += " " * space
        if j < len(scores):
            str_buffer += "\n\n"
    str_buffer += "\n" * newlines
    if return_str:
        return str_buffer
    else:
        print(str_buffer)
