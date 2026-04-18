import numpy as np


# ==============================
# HADA WEIGHT CALCULATION (FIXED)
# ==============================
# def compute_hada_weights(shap_scores, epsilons, tau=0.5, beta=1e-5):

#     shap_scores = np.array(shap_scores)
#     epsilons = np.array(epsilons)

#     # normalize SHAP
#     shap_scores = (shap_scores - np.min(shap_scores)) / (
#         np.max(shap_scores) - np.min(shap_scores) + 1e-8
#     )

#     # 🔥 add small variation boost
#     weights = shap_scores / (epsilons + beta)

#     # 🔥 mild exponential (not too strong)
#     weights = np.exp(tau * weights)

#     # normalize
#     weights = weights / np.sum(weights)

#     return weights
import numpy as np

def compute_hada_weights(shap_scores, epsilons, energies, tau=1.0, beta=1e-5):

    shap_scores = np.array(shap_scores)
    epsilons = np.array(epsilons)
    energies = np.array(energies)

    # ==============================
    # NORMALIZE SHAP
    # ==============================
    shap_scores = (shap_scores - np.min(shap_scores)) / (
        np.max(shap_scores) - np.min(shap_scores) + 1e-8
    )

    # ==============================
    # NORMALIZE ENERGY
    # ==============================
    energies = (energies - np.min(energies)) / (
        np.max(energies) - np.min(energies) + 1e-8
    )

    # ==============================
    # 🔥 FIX: SOFT ENERGY INFLUENCE
    # ==============================
    # convert energy to "efficiency"
    energy_eff = 1 / (energies + 1e-6)

    # normalize again
    energy_eff = (energy_eff - np.min(energy_eff)) / (
        np.max(energy_eff) - np.min(energy_eff) + 1e-8
    )

    # ==============================
    # 🔥 BETTER COMBINATION
    # ==============================
    # SHAP dominates, energy assists
    weights = (0.7 * shap_scores + 0.3 * energy_eff)

    # ==============================
    # PRIVACY ADJUSTMENT
    # ==============================
    weights = weights / (epsilons + beta)

    # ==============================
    # STABILIZATION (SOFTEN SHARPNESS)
    # ==============================
    weights = np.power(weights, 1.5)

    # ==============================
    # SOFTMAX NORMALIZATION
    # ==============================
    weights = np.exp(tau * weights)
    weights = weights / np.sum(weights)

    return weights

# ==============================
# HADA AGGREGATION
# ==============================
def hada_aggregation(local_weights, shap_scores, epsilons, energies):

    weights = compute_hada_weights(shap_scores, epsilons, energies)

    global_weights = np.zeros_like(local_weights[0])

    for w, lw in zip(weights, local_weights):
        global_weights += w * lw

    return global_weights


# ==============================
# APPLY GLOBAL MODEL (SIMPLIFIED)
# ==============================
def update_global_model(base_model, global_weights):
    return base_model