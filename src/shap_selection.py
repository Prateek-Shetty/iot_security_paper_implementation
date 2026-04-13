import shap
import numpy as np

# SHAP FEATURE SELECTION (paper method)
def select_features(model, X, num_features=30):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # mean importance
    importance = np.abs(shap_values).mean(axis=0)

    # select top features
    indices = np.argsort(importance)[-num_features:]

    return indices