import numpy as np
from src.client import train_client, get_model_weights
from src.shap_selection import compute_shap_values, compute_shap_stability
from src.dp import apply_differential_privacy
from src.server import hada_aggregation


# ==============================
# FEDERATED TRAINING LOOP (PAPER)
# ==============================
def federated_training(clients, rounds=5, epsilon=1.0):
    """
    Federated Learning with:
    - Local training (LightGBM)
    - SHAP-based stability (paper)
    - Differential Privacy (paper)
    - HADA aggregation (paper)

    Returns:
    - list of local models (used for ensemble prediction)
    """

    print("\n🚀 Starting Federated Learning (FL + SHAP + DP)...")

    local_models = []  # 🔥 keep outside loop (important)

    for r in range(rounds):
        print(f"\n🔁 Round {r+1}/{rounds}")

        local_models = []
        local_weights = []
        shap_scores = []
        epsilons = []

        # ==============================
        # CLIENT-SIDE COMPUTATION (PAPER)
        # ==============================
        for client in clients:
            X = client["X"]
            y = client["y"]

            # 1. Local training (LightGBM)
            model = train_client(X, y)
            local_models.append(model)

            # 2. Extract weights (proxy)
            weights = get_model_weights(model)

            # ==============================
            # 3. SHAP (PAPER)
            # ==============================
            shap_values = compute_shap_values(model, X[:1000])

            # 🔥 FIX: SHAP sometimes returns list → handle properly
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            stability = compute_shap_stability(shap_values)

            # ==============================
            # 4. DIFFERENTIAL PRIVACY (PAPER)
            # ==============================
            weights = apply_differential_privacy([weights], epsilon)[0]

            local_weights.append(weights)
            shap_scores.append(stability)
            epsilons.append(epsilon)

        # ==============================
        # 5. HADA AGGREGATION (PAPER CORE)
        # ==============================
        global_weights = hada_aggregation(local_weights, shap_scores, epsilons)

        print("✅ HADA aggregation complete")

        # NOTE:
        # LightGBM doesn't support weight injection
        # so we use ensemble later

    print("\n🎉 Federated Learning Completed")

    return local_models


# ==============================
# FEDERATED PREDICTION (FIXED)
# ==============================
def federated_predict(models, X):
    """
    Combine predictions from all client models
    (correct handling for binary classification)
    """

    probs = []

    for model in models:
        p = model.predict_proba(X)

        # 🔥 FIX: handle binary case properly
        if p.shape[1] == 2:
            probs.append(p[:, 1])  # probability of class 1
        else:
            probs.append(p.ravel())

    # average probabilities
    avg_prob = np.mean(probs, axis=0)

    # threshold for binary classification
    return (avg_prob > 0.5).astype(int)