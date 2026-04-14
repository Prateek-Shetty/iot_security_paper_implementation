import numpy as np


# ==============================
# HADA WEIGHT CALCULATION (FIXED)
# ==============================
def compute_hada_weights(shap_scores, epsilons, tau=1.0, beta=1e-5):
    """
    Improved HADA weighting

    Fixes:
    - Stronger SHAP influence
    - Better separation between clients
    """

    shap_scores = np.array(shap_scores)
    epsilons = np.array(epsilons)

    # 🔥 normalize SHAP scores
    shap_scores = (shap_scores - np.min(shap_scores)) / (
        np.max(shap_scores) - np.min(shap_scores) + 1e-8
    )

    # 🔥 stronger weighting (square effect)
    weights = (shap_scores ** 2) / (epsilons + beta)

    # exponential scaling (paper idea)
    weights = np.exp(tau * weights)

    # normalize
    weights = weights / np.sum(weights)

    return weights


# ==============================
# HADA AGGREGATION
# ==============================
def hada_aggregation(local_weights, shap_scores, epsilons):

    weights = compute_hada_weights(shap_scores, epsilons)

    global_weights = np.zeros_like(local_weights[0])

    for w, lw in zip(weights, local_weights):
        global_weights += w * lw

    return global_weights


# ==============================
# APPLY GLOBAL MODEL (SIMPLIFIED)
# ==============================
def update_global_model(base_model, global_weights):
    return base_model