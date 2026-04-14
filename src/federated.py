import numpy as np
from src.client import train_client, get_model_weights
from src.shap_selection import compute_shap_values, compute_shap_stability
from src.dp import apply_differential_privacy
from src.server import hada_aggregation, compute_hada_weights  # 🔥 added


# ==============================
# FEDERATED TRAINING LOOP (PAPER)
# ==============================
def federated_training(clients, rounds=5, epsilon=1.0):

    print("\n🚀 Starting Federated Learning (FL + SHAP + DP)...")

    local_models = []

    for r in range(rounds):
        print(f"\n🔁 Round {r+1}/{rounds}")

        local_models = []
        local_weights = []
        shap_scores = []
        epsilons = []

        for client in clients:
            X = client["X"]
            y = client["y"]

            model = train_client(X, y)
            local_models.append(model)

            weights = get_model_weights(model)

            # SHAP
            shap_values = compute_shap_values(model, X[:1000])

            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            stability = compute_shap_stability(shap_values)

            # DP
            weights = apply_differential_privacy([weights], epsilon)[0]

            local_weights.append(weights)
            shap_scores.append(stability)
            epsilons.append(epsilon)

        # HADA aggregation
        global_weights = hada_aggregation(local_weights, shap_scores, epsilons)

        print("✅ HADA aggregation complete")

    print("\n🎉 Federated Learning Completed")

    # 🔥 FIX: return shap + epsilon also
    return local_models, shap_scores, epsilons


# ==============================
# FEDERATED PREDICTION (FIXED)
# ==============================
def federated_predict(models, X, shap_scores=None, epsilons=None):

    probs = []

    # ==============================
    # NORMAL FL (no weighting)
    # ==============================
    if shap_scores is None or epsilons is None:
        for model in models:
            p = model.predict_proba(X)

            if p.shape[1] == 2:
                probs.append(p[:, 1])
            else:
                probs.append(p.ravel())

        avg_prob = np.mean(probs, axis=0)
        return (avg_prob > 0.5).astype(int)

    # ==============================
    # 🔥 HADA WEIGHTED PREDICTION
    # ==============================
    weights = compute_hada_weights(shap_scores, epsilons)

    weighted_probs = np.zeros(len(X))

    for w, model in zip(weights, models):
        p = model.predict_proba(X)

        if p.shape[1] == 2:
            weighted_probs += w * p[:, 1]
        else:
            weighted_probs += w * p.ravel()

    return (weighted_probs > 0.5).astype(int)