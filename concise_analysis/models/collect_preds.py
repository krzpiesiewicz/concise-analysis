import numpy as np
import pandas as pd


class ClassificationsCollector:
    def __init__(self, classes_order=None):
        self.res_dct = None
        self.res_proba_dct = None
        self.classes_preds = (
            {} if classes_order is None else {klass: 0 for klass in classes_order}
        )
        self.classes_preds_probas = (
            {} if classes_order is None else {klass: 0 for klass in classes_order}
        )
        self.classes_order = classes_order
        self.res_df = None
        self.res_proba_df = None
        self.is_res_up_to_date = False
        self.is_res_proba_up_to_date = False

    def __str__(self):
        if self.res_proba_dct is not None:
            return str(self.res_proba_dct)
        if self.res.dct is not None:
            return str(self.res_proba_dct)
        return str({})

    def add_preds(self, y_index, y_pred=None, y_pred_proba=None):

        if y_pred is not None:
            for idx, pred in zip(y_index, y_pred):
                if pred in self.classes_preds:
                    self.classes_preds[pred] += 1
                else:
                    self.classes_preds[pred] = 1
                if self.res_dct is None:
                    self.res_dct = {}
                if idx in self.res_dct:
                    dct = self.res_dct[idx]
                else:
                    dct = {"all_preds": 0, "preds": {}}
                    self.res_dct[idx] = dct
                dct["all_preds"] += 1
                if pred in dct["preds"]:
                    dct["preds"][pred] += 1
                else:
                    dct["preds"][pred] = 1

            self.is_res_up_to_date = False

        if y_pred_proba is not None:

            if type(y_pred_proba) is pd.DataFrame:
                classes_order = y_pred_proba.columns
                if self.classes_order is None:
                    self.classes_order = classes_order
                preds_perm = np.array(
                    [
                        np.where(classes_order == klass)[0][0]
                        for klass in self.classes_order
                    ]
                )
                y_pred_proba = y_pred_proba.values[preds_perm]
            if type(y_pred_proba) is pd.Series:
                y_pred_proba = y_pred_proba.values
            if type(y_pred_proba) is list:
                y_pred_proba = np.array(y_pred_proba)
            if len(y_pred_proba.shape) == 1:
                y_pred_proba = y_pred_proba.reshape((y_pred_proba.shape[0], 1))
            if y_pred_proba.shape[1] == 1:
                if self.classes_order is not None and len(self.classes_order) != 2:
                    raise Exception(
                        f"pd.Series passed as y_pred_proba but classification is nonbinary "
                        + f"(classes: {self.classes_order})"
                    )
                ones_proba = y_pred_proba
                y_pred_proba = np.zeros((ones_proba.shape[0], 2))
                y_pred_proba[:, 0] = 1 - ones_proba[:, 0]
                y_pred_proba[:, 1] = ones_proba[:, 0]
            if self.classes_order is None:
                self.classes_order = [i for i in range(y_pred_proba.shape[1])]

            for idx, pred_proba in zip(y_index, y_pred_proba):
                if self.res_proba_dct is None:
                    self.res_proba_dct = {}
                if idx in self.res_proba_dct:
                    dct = self.res_proba_dct[idx]
                else:
                    dct = {"all_preds": 0, "probas": {}}
                    self.res_proba_dct[idx] = dct
                dct["all_preds"] += 1
                for klass, proba in zip(self.classes_order, pred_proba):
                    if klass in dct["probas"]:
                        dct["probas"][klass] += proba
                    else:
                        dct["probas"][klass] = proba
                    if klass in self.classes_preds_probas:
                        self.classes_preds_probas[klass] += proba
                    else:
                        self.classes_preds_probas[klass] = proba

            self.is_res_proba_up_to_date = False

    def __update_res_dfs__(self):
        if not self.is_res_up_to_date and self.res_dct is not None:
            dct = self.res_dct
            classes = (
                self.classes_order
                if self.classes_order is not None
                else list(self.classes_preds.keys())
            )
            index = pd.Index(list(dct.keys()))
            records = [
                tuple(
                    [
                        dct[idx]["preds"][klass] if klass in dct[idx]["preds"] else 0
                        for klass in classes
                    ]
                    + [dct[idx]["all_preds"]]
                )
                for idx in index
            ]
            self.res_df = pd.DataFrame.from_records(
                records, columns=classes + ["all_preds"], index=index
            ).sort_index()

            df = self.res_df.copy()
            for klass in classes:
                df[klass] = df[klass] / df["all_preds"]
            self.res_ratios_df = df

        self.is_res_up_to_date = True

    def __update_res_proba_dfs__(self):
        if not self.is_res_proba_up_to_date and self.res_proba_dct is not None:
            dct = self.res_proba_dct
            classes = (
                self.classes_order
                if self.classes_order is not None
                else list(self.classes_preds_proba.keys())
            )
            index = pd.Index(list(dct.keys()))
            records = [
                tuple(
                    [
                        dct[idx]["probas"][klass] if klass in dct[idx]["probas"] else 0
                        for klass in classes
                    ]
                    + [dct[idx]["all_preds"]]
                )
                for idx in index
            ]
            self.res_proba_df = pd.DataFrame.from_records(
                records, columns=classes + ["all_preds"], index=index
            ).sort_index()
            for klass in classes:
                self.res_proba_df[klass] = (
                    self.res_proba_df[klass] / self.res_proba_df["all_preds"]
                )

        self.is_res_proba_up_to_date = True

    def get_preds_and_proba(self, index=None):
        self.__update_res_dfs__()
        self.__update_res_proba_dfs__()

        res_df = (
            self.res_proba_df if self.res_proba_df is not None else self.res_ratios_df
        )
        if index is not None:
            res_df = res_df.loc[index]
        if res_df is not None:
            classes = (
                self.classes_order
                if self.classes_order is not None
                else list(self.classes_preds.keys())
            )
            preds = pd.DataFrame([], index=res_df.index, columns=["pred", "proba"])
            for idx, row in res_df.iterrows():
                best_proba = 0
                best_pred = None
                for klass in classes:
                    proba = row[klass]
                    if proba > best_proba:
                        best_pred = klass
                        best_proba = proba
                preds.loc[idx, "pred"] = best_pred
                preds.loc[idx, "proba"] = best_proba
            return preds
        else:
            return None

    def get_results(self, ratios=True):
        self.__update_res_dfs__()
        if ratios:
            return self.res_ratios_df
        else:
            return self.res_df

    def get_proba_results(self):
        self.__update_res_proba_dfs__()
        return self.res_proba_df


class RegressionPredictionsCollector:
    def __init__(self):
        self.res_dct = dict()
        self.res_df = None
        self.is_res_up_to_date = False

    def __str__(self):
        return str(self.res_dct)

    def add_preds(self, y_index, y_pred):
        for idx, pred in zip(y_index, y_pred):
            if idx in self.res_dct:
                preds_lst = self.res_dct[idx]
            else:
                preds_lst = []
                self.res_dct[idx] = preds_lst
            preds_lst.append(pred)
        self.is_res_up_to_date = False

    def __update_res_dfs__(self):
        if not self.is_res_up_to_date:
            self.res_df = pd.DataFrame(
                [], index=pd.Index(sorted(self.res_dct.keys())), columns=["pred", "std"]
            )
            for idx, preds_lst in self.res_dct.items():
                self.res_df.loc[idx, "pred"] = np.mean(preds_lst)
                self.res_df.loc[idx, "std"] = np.std(preds_lst)
            self.is_res_up_to_date = True

    def get_results(self):
        self.__update_res_dfs__()
        return self.res_df


def collect_ensemble_predictions(models, X, pred_collector):
    for model in models:
        y_pred = model.predict(X)
        pred_collector.add_preds(X.index, y_pred)
