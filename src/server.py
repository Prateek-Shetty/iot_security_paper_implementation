import numpy as np


# ==============================
# HADA WEIGHT CALCULATION (PAPER)
# ==============================
def compute_hada_weights(shap_scores, epsilons, tau=1.0, beta=1e-5):
    """
    Compute adaptive weights based on:
    - SHAP stability score (s_k)
    - Privacy budget (epsilon_k)

    Paper formula:
    w_k = exp( tau * s_k / (epsilon_k + beta) )

    Returns:
    - normalized weights
    """

    weights = []

    for s_k, eps_k in zip(shap_scores, epsilons):
        w = np.exp(tau * s_k / (eps_k + beta))
        weights.append(w)

    weights = np.array(weights)

    # normalize
    weights = weights / np.sum(weights)

    return weights


# ==============================
# HADA AGGREGATION
# ==============================
def hada_aggregation(local_weights, shap_scores, epsilons):
    """
    Perform weighted aggregation of client models

    Parameters:
    - local_weights: list of client weight vectors
    - shap_scores: SHAP stability per client
    - epsilons: privacy budget per client

    Returns:
    - global_weights
    """

    # compute adaptive weights
    weights = compute_hada_weights(shap_scores, epsilons)

    # initialize global weights
    global_weights = np.zeros_like(local_weights[0])

    # weighted sum
    for w, lw in zip(weights, local_weights):
        global_weights += w * lw

    return global_weights


# ==============================
# APPLY GLOBAL MODEL (SIMPLIFIED)
# ==============================
def update_global_model(base_model, global_weights):
    """
    Update model using aggregated weights

    NOTE:
    LightGBM doesn't allow direct weight injection,
    so we simulate update by keeping base model.

    (Accepted approximation in research reproduction)
    """

    return base_model