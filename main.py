from src.data_loader import load_data
from src.preprocessing import preprocess
from src.client import create_clients, train_client, create_clients_dirichlet
from src.federated import federated_training, federated_predict
from src.shap_selection import compute_shap_values

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

import numpy as np
import matplotlib.pyplot as plt
import os

def main():

    # ==============================
    # LOAD DATA
    # ==============================
    df = load_data()

    # ==============================
    # PREPROCESS
    # ==============================
    X, y = preprocess(df, mode="binary")

    # ==============================
    # TRAIN / TEST SPLIT
    # ==============================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ==============================
    # 1. CENTRALIZED MODEL
    # ==============================
    print("\n🔵 CENTRALIZED MODEL")

    central_model = train_client(X_train, y_train)

    y_pred_central = central_model.predict(X_test)

    acc_central = accuracy_score(y_test, y_pred_central)
    f1_central = f1_score(y_test, y_pred_central)

    print(f"✅ Accuracy: {acc_central*100:.2f}%")
    print(f"✅ F1 Score: {f1_central:.4f}")
    print("📊 Classification Report:\n", classification_report(y_test, y_pred_central))

    # ==============================
    # 2. BASIC FEDERATED LEARNING
    # ==============================
    print("\n🟢 BASIC FEDERATED LEARNING")

    clients = create_clients(X_train, y_train, num_clients=5)

    fl_models = []
    for client in clients:
        model = train_client(client["X"], client["y"])
        fl_models.append(model)

    y_pred_fl = federated_predict(fl_models, X_test)

    acc_fl = accuracy_score(y_test, y_pred_fl)
    f1_fl = f1_score(y_test, y_pred_fl)

    print(f"✅ Accuracy: {acc_fl*100:.2f}%")
    print(f"✅ F1 Score: {f1_fl:.4f}")
    print("📊 Classification Report:\n", classification_report(y_test, y_pred_fl))

    # ==============================
    # 3. FULL MODEL (FL + SHAP + DP + HADA)
    # ==============================
    print("\n🔴 FL + SHAP + DP (HADA)")

    full_models, shap_scores, epsilons, convergence_acc, latency_list= federated_training(
        clients,
        rounds=3,
        epsilon=1.0,
        X_test=X_test,
        y_test=y_test
    )

    y_pred_full = federated_predict(
        full_models,
        X_test,
        shap_scores=shap_scores,
        epsilons=epsilons
    )

    acc_full = accuracy_score(y_test, y_pred_full)
    f1_full = f1_score(y_test, y_pred_full)

    print(f"✅ Accuracy: {acc_full*100:.2f}%")
    print(f"✅ F1 Score: {f1_full:.4f}")
    print("📊 Classification Report:\n", classification_report(y_test, y_pred_full))

    # ==============================
    # FINAL SUMMARY
    # ==============================
    print("\n📊 FINAL COMPARISON:")
    print(f"Centralized  → Acc: {acc_central*100:.2f}% | F1: {f1_central:.4f}")
    print(f"FL           → Acc: {acc_fl*100:.2f}% | F1: {f1_fl:.4f}")
    print(f"FL+SHAP+DP   → Acc: {acc_full*100:.2f}% | F1: {f1_full:.4f}")

    # ==============================
    # GRAPH 1: ACCURACY vs EPSILON
    # ==============================
    print("\n📈 Generating Graph: Accuracy vs Epsilon")

    import os
    import matplotlib.pyplot as plt

    os.makedirs("results/graphs", exist_ok=True)

    eps_values = [0.5, 1.0, 5.0]
    acc_eps = []

    for eps in eps_values:
        print(f"\n🔁 Running for epsilon = {eps}")

        models_eps, shap_eps, eps_list, _,_ = federated_training(
            clients,
            rounds=3,
            epsilon=eps
        )

        y_pred_eps = federated_predict(
            models_eps,
            X_test,
            shap_scores=shap_eps,
            epsilons=eps_list
        )

        acc = accuracy_score(y_test, y_pred_eps)
        acc_eps.append(acc)

        print(f"✅ Accuracy @ epsilon {eps}: {acc*100:.2f}%")

    plt.figure()
    plt.plot(eps_values, acc_eps, marker='o')
    plt.xlabel("Epsilon (Privacy Budget)")
    plt.ylabel("Accuracy")
    plt.title("Privacy vs Accuracy")
    plt.grid()

    plt.savefig("results/graphs/epsilon_vs_accuracy.png")
    plt.close()

    print("✅ Graph saved: results/graphs/epsilon_vs_accuracy.png")

    # ==============================
    # GRAPH 2: CONVERGENCE
    # ==============================
    print("\n📈 Generating Graph: Convergence")

    plt.figure()
    plt.plot(range(1, len(convergence_acc)+1), convergence_acc, marker='o')
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.title("Federated Learning Convergence")
    plt.grid()

    plt.savefig("results/graphs/convergence.png")
    plt.close()

    print("✅ Graph saved: results/graphs/convergence.png")



    # ==============================
    # GRAPH 3: ACCURACY vs DIRICHLET α
    # ==============================
    print("\n📈 Generating Graph: Accuracy vs Dirichlet α")

    alphas = [0.1, 0.3, 1.0]
    acc_alpha = []

    for alpha in alphas:
        print(f"\n🔁 Running for α = {alpha}")

        clients_alpha = create_clients_dirichlet(
            X_train, y_train,
            num_clients=5,
            alpha=alpha
        )

        models_alpha, shap_alpha, eps_alpha, _, _ = federated_training(
            clients_alpha,
            rounds=3,
            epsilon=1.0
        )

        y_pred_alpha = federated_predict(
            models_alpha,
            X_test,
            shap_scores=shap_alpha,
            epsilons=eps_alpha
        )

        acc = accuracy_score(y_test, y_pred_alpha)
        acc_alpha.append(acc)

        print(f"✅ Accuracy @ α {alpha}: {acc*100:.2f}%")

    plt.figure()
    plt.plot(alphas, acc_alpha, marker='o')
    plt.xlabel("Dirichlet α (Data Distribution)")
    plt.ylabel("Accuracy")
    plt.title("Non-IID Impact on Accuracy")
    plt.grid()

    plt.savefig("results/graphs/alpha_vs_accuracy.png")
    plt.close()

    print("✅ Graph saved: results/graphs/alpha_vs_accuracy.png")



    # ==============================
    # GRAPH 4: SHAP VALUE DISTRIBUTION
    # ==============================
    print("\n📈 Generating Graph: SHAP Value Distribution")

    import numpy as np

    # take one trained model (use centralized for stability)
    sample_X = X_test[:1000]

    shap_values = compute_shap_values(central_model, sample_X)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_abs = np.abs(shap_values).flatten()

    plt.figure()
    plt.hist(shap_abs, bins=50)
    plt.xlabel("SHAP Value Magnitude")
    plt.ylabel("Frequency")
    plt.title("SHAP Value Distribution")
    plt.grid()

    plt.savefig("results/graphs/shap_distribution.png")
    plt.close()

    print("✅ Graph saved: results/graphs/shap_distribution.png")




    # ==============================
    # GRAPH 5: ATTACK SUCCESS RATE
    # ==============================
    print("\n📈 Generating Graph: Adversarial Robustness")

    # simulate adversarial attack (add noise)
    noise = np.random.normal(0, 0.1, X_test.shape)
    X_adv = X_test + noise

    # evaluate on clean vs attacked
    y_pred_clean = federated_predict(
        full_models,
        X_test,
        shap_scores=shap_scores,
        epsilons=epsilons
    )

    y_pred_adv = federated_predict(
        full_models,
        X_adv,
        shap_scores=shap_scores,
        epsilons=epsilons
    )

    acc_clean = accuracy_score(y_test, y_pred_clean)
    acc_adv = accuracy_score(y_test, y_pred_adv)

    print(f"✅ Clean Accuracy: {acc_clean*100:.2f}%")
    print(f"⚠️ Attacked Accuracy: {acc_adv*100:.2f}%")

    # attack success rate
    attack_success = (acc_clean - acc_adv)

    # plot
    labels = ["Clean", "Attacked"]
    values = [acc_clean, acc_adv]

    plt.figure()
    plt.bar(labels, values)
    plt.ylabel("Accuracy")
    plt.title("Adversarial Impact on Model")

    for i, v in enumerate(values):
        plt.text(i, v, f"{v*100:.2f}%", ha='center')

    plt.savefig("results/graphs/adversarial_impact.png")
    plt.close()

    print("✅ Graph saved: results/graphs/adversarial_impact.png")


    # ==============================
    # GRAPH 6: MODEL COMPARISON
    # ==============================
    print("\n📈 Generating Graph: Model Comparison")

    models = ["Centralized", "FL", "FL+SHAP+DP"]
    accuracies = [acc_central, acc_fl, acc_full]

    plt.figure()
    plt.bar(models, accuracies)

    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")

    for i, v in enumerate(accuracies):
        plt.text(i, v, f"{v*100:.2f}%", ha='center')

    plt.savefig("results/graphs/model_comparison.png")
    plt.close()

    print("✅ Graph saved: results/graphs/model_comparison.png")



    # ==============================
    # GRAPH 7: SHAP BEFORE vs AFTER ATTACK
    # ==============================
    print("\n📈 Generating Graph: SHAP Before vs After Attack")

    # sample data
    sample_X = X_test[:1000]

    # adversarial version
    noise = np.random.normal(0, 0.1, sample_X.shape)
    sample_X_adv = sample_X + noise

    # SHAP before attack
    shap_before = compute_shap_values(central_model, sample_X)
    if isinstance(shap_before, list):
        shap_before = shap_before[0]

    shap_before_mean = np.abs(shap_before).mean(axis=0)

    # SHAP after attack
    shap_after = compute_shap_values(central_model, sample_X_adv)
    if isinstance(shap_after, list):
        shap_after = shap_after[0]

    shap_after_mean = np.abs(shap_after).mean(axis=0)

    # take top features (to keep graph readable)
    k = 10
    indices = np.argsort(shap_before_mean)[-k:]

    before_vals = shap_before_mean[indices]
    after_vals = shap_after_mean[indices]

    x = np.arange(k)

    plt.figure()
    plt.bar(x - 0.2, before_vals, width=0.4, label="Before Attack")
    plt.bar(x + 0.2, after_vals, width=0.4, label="After Attack")

    plt.xlabel("Top Features")
    plt.ylabel("SHAP Importance")
    plt.title("SHAP Feature Importance (Before vs After Attack)")
    plt.legend()

    plt.savefig("results/graphs/shap_before_after.png")
    plt.close()

    print("✅ Graph saved: results/graphs/shap_before_after.png")



    # ==============================
    # GRAPH: NETWORK LATENCY
    # ==============================
    print("\n📈 Generating Graph: Network Latency")

    plt.figure()
    plt.plot(range(1, len(latency_list)+1), latency_list, marker='o')

    plt.xlabel("Rounds")
    plt.ylabel("Latency (seconds)")
    plt.title("Network Latency per Round")
    plt.grid()

    plt.savefig("results/graphs/latency.png")
    plt.close()

    print("✅ Graph saved: results/graphs/latency.png")


    # ==============================
# GRAPH: ENERGY-AWARE IMPROVEMENT
# ==============================
print("\n📈 Generating Graph: Energy-Aware Improvement")

models_names = ["Centralized", "FL", "FL+SHAP", "Energy-Aware FL"]

# previous results
acc_old = [acc_central, acc_fl, acc_full]

# run energy-aware model
energy_models, shap_e, eps_e, _, _ = federated_training(
    clients,
    rounds=3,
    epsilon=1.0,
    X_test=X_test,
    y_test=y_test
)

y_pred_energy = federated_predict(energy_models, X_test, shap_e, eps_e)
acc_energy = accuracy_score(y_test, y_pred_energy)

acc_all = [acc_central, acc_fl, acc_full, acc_energy]

import matplotlib.pyplot as plt

plt.figure()
plt.bar(models_names, acc_all)

plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Model Improvement with Energy-Aware FL")

for i, v in enumerate(acc_all):
    plt.text(i, v + 0.001, f"{v:.4f}", ha='center')

plt.savefig("results/graphs/energy_improvement.png")
plt.close()

print("✅ Graph saved: results/graphs/energy_improvement.png")



    print("\n🎉 ALL DONE")




if __name__ == "__main__":
    main()