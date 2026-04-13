import numpy as np

# HADA-LIKE WEIGHTED AGGREGATION (paper concept)
def aggregate_models(models, weights=None):
    # extract model parameters
    model_weights = [model.booster_.feature_importance() for model in models]

    model_weights = np.array(model_weights)

    if weights is None:
        weights = np.ones(len(models)) / len(models)

    # weighted aggregation
    global_weights = np.average(model_weights, axis=0, weights=weights)

    return global_weights