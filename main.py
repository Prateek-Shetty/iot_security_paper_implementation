from src.data_loader import load_data
from src.preprocessing import preprocess
from src.client import create_clients, train_client
from src.federated import federated_training, federated_predict

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report


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

    full_models, shap_scores, epsilons = federated_training(
        clients,
        rounds=3,   # 🔥 reduced rounds (stable)
        epsilon=1.0
    )

    y_pred_full = federated_predict(full_models, X_test, shap_scores=shap_scores, epsilons=epsilons)

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

    print("\n🎉 ALL DONE")


if __name__ == "__main__":
    main()