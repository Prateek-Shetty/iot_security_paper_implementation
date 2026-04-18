import numpy as np
import time

from src.client import train_client, get_model_weights
from src.shap_selection import compute_shap_values, compute_shap_stability
from src.dp import apply_differential_privacy
from src.server import hada_aggregation, compute_hada_weights


# ==============================
# CLIENT SELECTION (ENERGY-AWARE)
# ==============================
def select_top_clients(clients, fraction=0.7):
    """
    Select most energy-efficient clients
    (lower energy = better)
    """

    num_selected = max(1, int(len(clients) * fraction))

    sorted_clients = sorted(
        clients,
        key=lambda c: c.get("energy", 1.0)
    )

    return sorted_clients[:num_selected]


# ==============================
# FEDERATED TRAINING (OPTIMIZED)
# ==============================
def federated_training(clients, rounds=5, epsilon=1.0, X_test=None, y_test=None):

    print("\n🚀 Starting Federated Learning (FL + SHAP + DP + ENERGY)...")

    convergence_acc = []
    latency_list = []

    # 🔥 convergence tracking
    prev_acc = 0
    stagnant_rounds = 0

    for r in range(rounds):

        print(f"\n🔁 Round {r+1}/{rounds}")
        round_start = time.time()

        local_models = []
        local_weights = []
        shap_scores = []
        epsilons = []
        energies = []

        # ==============================
        # 🔥 CLIENT SELECTION (NEW)
        # ==============================
        selected_clients = select_top_clients(clients, fraction=0.8)

        print(f"👥 Selected {len(selected_clients)}/{len(clients)} clients")

        for client in selected_clients:

            X = client["X"]
            y = client["y"]

            # ==============================
            # LOCAL TRAINING (with early stopping)
            # ==============================
            model = train_client(X, y)
            local_models.append(model)

            # energy tracking
            energy = client.get("energy", 1.0)
            energies.append(energy)

            # ==============================
            # MODEL WEIGHTS
            # ==============================
            weights = get_model_weights(model)

            # ==============================
            # SHAP (OPTIMIZED)
            # ==============================
            shap_values = compute_shap_values(
                model,
                X,
                current_round=r
            )

            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            stability = compute_shap_stability(shap_values)

            # ==============================
            # DIFFERENTIAL PRIVACY
            # ==============================
            weights = apply_differential_privacy([weights], epsilon)[0]

            # collect
            local_weights.append(weights)
            shap_scores.append(stability)
            epsilons.append(epsilon)

        # ==============================
        # SIMULATED LATENCY
        # ==============================
        time.sleep(0.05 * len(selected_clients))

        if len(local_models) == 0:
            print("⚠️ No clients available")
            continue

        # ==============================
        # 🔥 HADA AGGREGATION (ENERGY-AWARE)
        # ==============================
        global_weights = hada_aggregation(
            local_weights,
            shap_scores,
            epsilons,
            energies
        )

        round_time = time.time() - round_start
        latency_list.append(round_time)

        print(f"⏱️ Round Latency: {round_time:.3f} sec")
        print("✅ Aggregation complete")

        # ==============================
        # CONVERGENCE TRACKING
        # ==============================
        if X_test is not None and y_test is not None:

            from sklearn.metrics import accuracy_score

            preds = federated_predict(
                local_models,
                X_test,
                shap_scores,
                epsilons,
                energies
            )

            acc = accuracy_score(y_test, preds)
            convergence_acc.append(acc)

            print(f"📈 Round {r+1} Accuracy: {acc*100:.2f}%")

            # 🔥 EARLY STOPPING (NEW)
            if abs(acc - prev_acc) < 0.001:
                stagnant_rounds += 1
            else:
                stagnant_rounds = 0

            if stagnant_rounds >= 3:
                print("🛑 Early convergence reached. Stopping training.")
                break

            prev_acc = acc

    print("\n🎉 Federated Learning Completed")

    return local_models, shap_scores, epsilons, convergence_acc, latency_list, energies


# ==============================
# FEDERATED PREDICTION (FIXED)
# ==============================
def federated_predict(models, X, shap_scores=None, epsilons=None, energies=None):

    probs = []

    # ==============================
    # NORMAL FL
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
    # SAFETY FIX
    # ==============================
    if energies is None:
        energies = np.ones(len(shap_scores))

    # ==============================
    # 🔥 ENERGY-AWARE WEIGHTING
    # ==============================
    weights = compute_hada_weights(shap_scores, epsilons, energies)

    weighted_probs = np.zeros(len(X))

    for w, model in zip(weights, models):

        p = model.predict_proba(X)

        if p.shape[1] == 2:
            weighted_probs += w * p[:, 1]
        else:
            weighted_probs += w * p.ravel()

    return (weighted_probs > 0.5).astype(int)