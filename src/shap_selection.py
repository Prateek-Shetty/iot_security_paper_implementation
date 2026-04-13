import shap
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ==============================
# SHAP FEATURE IMPORTANCE (PAPER)
# ==============================
def compute_shap_values(model, X):
    """
    Compute SHAP values for a trained model

    Parameters:
    - model: trained LightGBM model
    - X: input data

    Returns:
    - shap_values
    """

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    return shap_values


# ==============================
# SHAP STABILITY SCORE (PAPER IDEA)
# ==============================
def compute_shap_stability(shap_values):
    """
    Compute stability score of SHAP values

    Paper concept:
    → stable features = reliable clients

    Returns:
    - stability_score (scalar)
    """

    # absolute importance
    shap_abs = np.abs(shap_values)

    # mean importance
    mean_importance = np.mean(shap_abs)

    return mean_importance


# ==============================
# FEATURE SELECTION (TOP-K)
# ==============================
def select_top_features(shap_values, k=30):
    """
    Select top-k important features based on SHAP

    Returns:
    - indices of selected features
    """

    shap_abs = np.abs(shap_values).mean(axis=0)

    indices = np.argsort(shap_abs)[-k:]

    return indices