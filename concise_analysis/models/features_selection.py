import numpy as np


def print_forests_importances(forests, names, newlines_begin=0):
    if type(forests) is not list:
        forests = [forests]
    importances = np.mean([model.feature_importances_ for model in forests], axis=0)
    indices = np.argsort(importances)[::-1]

    print("\n" * newlines_begin + "Random forest feature ranking:")

    importances_stds = np.std(
        [tree.feature_importances_ for model in forests for tree in model.estimators_],
        axis=0,
    )

    for i in range(len(names)):
        print(
            f"{i + 1:2}. import. {importances[indices[i]]:.3f} (std: {importances_stds[indices[i]]:.3f})"
            + f"  –  {names[indices[i]]}"
        )
    print("")
