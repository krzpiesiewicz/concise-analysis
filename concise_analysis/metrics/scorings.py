import sklearn.metrics
from concise_analysis.metrics.metrics import confusion_score, mase, rmse, \
    rmspe, auc

confusion_scoring = {
    "name": "confusion",
    "pred_type": "pred",
    "scoring": confusion_score,
    "res_type": object,
}

mse_scoring = {
    "name": "mse",
    "pred_type": "pred",
    "scoring": sklearn.metrics.mean_squared_error,
    "res_type": float,
}

rmse_scoring = {"name": "rmse", "pred_type": "pred", "scoring": rmse,
                "res_type": float}

rmspe_scoring = {"name": "rmspe", "pred_type": "pred", "scoring": rmspe,
                 "res_type": float}

mae_scoring = {
    "name": "mae",
    "pred_type": "pred",
    "scoring": sklearn.metrics.mean_absolute_error,
    "res_type": float,
}

mape_scoring = {
    "name": "mape",
    "pred_type": "pred",
    "scoring": sklearn.metrics.mean_absolute_percentage_error,
    "res_type": float,
}

mase_scoring = {"name": "mase", "pred_type": "pred", "scoring": mase,
                "res_type": float}

medae_scoring = {
    "name": "medae",
    "pred_type": "pred",
    "scoring": sklearn.metrics.median_absolute_error,
    "res_type": float,
}

r2_scoring = {
    "name": "r2",
    "pred_type": "pred",
    "scoring": sklearn.metrics.r2_score,
    "res_type": float,
}

bacc_scoring = {
    "name": "bacc",
    "pred_type": "pred",
    "scoring": sklearn.metrics.balanced_accuracy_score,
    "res_type": float,
}

acc_scoring = {
    "name": "acc",
    "pred_type": "pred",
    "scoring": sklearn.metrics.accuracy_score,
    "res_type": float,
}

auc_scoring = {
    "name": "auc",
    "pred_type": "proba",
    "scoring": auc,
    "res_type": float,
}

scoring_dct = dict(
    mse=mse_scoring,
    MSE=mse_scoring,
    rmse=rmse_scoring,
    RMSE=rmse_scoring,
    RMSPE=rmspe_scoring,
    rmspe=rmspe_scoring,
    mae=mae_scoring,
    MAE=mae_scoring,
    mape=mape_scoring,
    MAPE=mape_scoring,
    mase=mase_scoring,
    MASE=mase_scoring,
    medae=medae_scoring,
    MEDAE=medae_scoring,
    r2=r2_scoring,
    R2=r2_scoring,
    bacc=bacc_scoring,
    BACC=bacc_scoring,
    acc=acc_scoring,
    ACC=acc_scoring,
    auc=auc_scoring,
    AUC=auc_scoring,
    confusion=confusion_scoring,
    confusion_score=confusion_scoring,
    confusion_matrix=confusion_scoring,
)


def get_scoring(scoring):
    if type(scoring) is str:
        if scoring in scoring_dct:
            return scoring_dct[scoring].copy()
        else:
            raise Exception(
                f"Unknown scoring. Available ones are: {scoring_dct.keys()}"
            )
    elif type(scoring) is dict:
        return scoring
    else:
        raise Exception(f"Argument scoring has to be of type str or dict")
