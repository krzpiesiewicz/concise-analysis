import numpy as np


def forests_importances(forests, names=None):
    if type(forests) is not list:
        forests = [forests]
    importances = np.mean([model.feature_importances_ for model in forests],
                          axis=0)
    indices = np.argsort(importances)[::-1]

    importances_stds = np.std(
        [tree.feature_importances_ for model in forests for tree in
         model.estimators_],
        axis=0,
    )

    return importances, importances_stds, indices, names


def print_importances(
        importances,
        importances_stds,
        indices,
        names=None,
        model_name=None,
        max_ftrs=None,
        newlines_begin=0
):
    model_name = model_name + " " if model_name is not None else ""
    print("\n" * newlines_begin + f"{model_name}features ranking:")
    for i in range(min(len(importances), len(indices))):
        if max_ftrs is not None and i == max_ftrs:
            break
        print(
            f"{i + 1:2}. import. {importances[indices[i]]:.3f} (std: {importances_stds[indices[i]]:.3f})"
            + f"  –  {names[indices[i]]}" if names is not None else ""
        )
    print("")


def print_forests_importances(forests, names=None, max_ftrs=None,
                              newlines_begin=0):
    importances, importances_stds, indices, names = forests_importances(
        forests, names=names)
    print_importances(importances, importances_stds, indices, names,
                      max_ftrs=max_ftrs,
                      model_name="Random forest",
                      newlines_begin=newlines_begin)
