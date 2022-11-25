import numpy as np
import pandas as pd


class ClassificationsCollector:
    def __init__(self, classes_order):
        self.res_dct = dict()
        self.classes = (
            {} if classes_order is None else {klass: 0 for klass in classes_order}
        )
        self.classes_order = classes_order
        self.res_df = None
        self.is_res_up_to_date = False

    def __str__(self):
        return str(self.res_dct)

    def add_preds(self, y_index, y_pred):
        for idx, pred in zip(y_index, y_pred):
            if pred in self.classes:
                self.classes[pred] += 1
            else:
                self.classes[pred] = 1
            if idx in self.res_dct:
                dct = self.res_dct[idx]
            else:
                dct = {"all_preds": 0}
                self.res_dct[idx] = dct
            dct["all_preds"] += 1
            if pred in dct:
                dct[pred] += 1
            else:
                dct[pred] = 1
        self.is_res_up_to_date = False

    def __update_res_dfs__(self):
        if not self.is_res_up_to_date:
            dct = self.res_dct
            classes = (
                self.classes_order
                if self.classes_order is not None
                else list(self.classes.keys())
            )
            columns = classes + ["all_preds"]
            index = pd.Index(list(dct.keys()))
            records = [
                tuple(
                    [
                        dct[idx][column] if column in dct[idx] else 0
                        for column in columns
                    ]
                )
                for idx in index
            ]
            self.res_df = pd.DataFrame.from_records(
                records, columns=columns, index=index
            ).sort_index()
            self.is_res_up_to_date = True
            df = self.res_df.copy()
            for klass in classes:
                df.loc[:, klass] /= df["all_preds"]
            self.res_ratios_df = df
            self.is_res_up_to_date = True

    def get_preds_and_proba(self):
        self.__update_res_dfs__()
        classes = self.classes.keys()
        preds = pd.DataFrame(
            [], index=self.res_ratios_df.index, columns=["pred", "proba"]
        )
        for idx, row in self.res_ratios_df.iterrows():
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

    def get_results(self, ratios=True):
        self.__update_res_dfs__()
        if ratios:
            return self.res_ratios_df
        else:
            self.res_df


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
