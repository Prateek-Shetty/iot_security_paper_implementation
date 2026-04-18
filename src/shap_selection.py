import shap
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ==============================
# GLOBAL CACHE (NEW)
# ==============================
LAST_SHAP_VALUES = None
LAST_ROUND = -1


# ==============================
# SHAP FEATURE IMPORTANCE (OPTIMIZED)
# ==============================
def compute_shap_values(model, X, current_round=0, interval=5, sample_frac=0.3):
    """
    Optimized SHAP computation

    Improvements:
    ✔ Compute only every 'interval' rounds
    ✔ Use sampling to reduce cost
    ✔ Cache results

    Parameters:
    - model: trained LightGBM model
    - X: input data (pandas DataFrame)
    - current_round: FL round number
    - interval: SHAP recompute frequency
    - sample_frac: fraction of data for SHAP

    Returns:
    - shap_values
    """

    global LAST_SHAP_VALUES, LAST_ROUND

    # ------------------------------
    # SKIP COMPUTATION (CACHE HIT)
    # ------------------------------
    if LAST_SHAP_VALUES is not None and (current_round - LAST_ROUND) < interval:
        return LAST_SHAP_VALUES

    # ------------------------------
    # SAMPLE DATA (MAJOR SPEED BOOST)
    # ------------------------------
    if hasattr(X, "sample"):
        X_sample = X.sample(frac=sample_frac, random_state=42)
    else:
        # fallback if numpy array
        n = int(len(X) * sample_frac)
        idx = np.random.choice(len(X), n, replace=False)
        X_sample = X[idx]

    # ------------------------------
    # SHAP COMPUTATION
    # ------------------------------
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # ------------------------------
    # CACHE UPDATE
    # ------------------------------
    LAST_SHAP_VALUES = shap_values
    LAST_ROUND = current_round

    return shap_values


# ==============================
# SHAP STABILITY SCORE (UNCHANGED + SAFE)
# ==============================
def compute_shap_stability(shap_values):
    """
    Stability score of SHAP values

    Returns:
    - stability_score (scalar)
    """

    shap_abs = np.abs(shap_values)

    # handle multi-class case safely
    if isinstance(shap_values, list):
        shap_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0)

    mean_importance = np.mean(shap_abs)

    return mean_importance


# ==============================
# FEATURE SELECTION (OPTIMIZED)
# ==============================
def select_top_features(shap_values, k=30):
    """
    Select top-k important features

    Improvements:
    ✔ Handles multi-class SHAP
    ✔ More stable ranking

    Returns:
    - indices of selected features
    """

    # handle multi-class SHAP output
    if isinstance(shap_values, list):
        shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        shap_values = np.abs(shap_values)

    shap_mean = shap_values.mean(axis=0)

    indices = np.argsort(shap_mean)[-k:]

    return indices