from src.data_loader import load_data
from src.preprocessing import preprocess
from src.client import create_clients, train_client
from src.federated import federated_training, federated_predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():

    # ==============================
    # LOAD DATA
    # ==============================
    df = load_data()

    # ==============================
    # PREPROCESS (PAPER SETUP)
    # ==============================
    X, y = preprocess(df, mode="binary")

    # ==============================
    # TRAIN-TEST SPLIT
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

    y_pred = central_model.predict(X_test)
    acc_central = accuracy_score(y_test, y_pred)

    print(f"✅ Centralized Accuracy: {acc_central*100:.2f}%")

    # ==============================
    # 2. BASIC FEDERATED LEARNING (NO SHAP, NO DP)
    # ==============================
    print("\n🟢 BASIC FEDERATED LEARNING")

    clients = create_clients(X_train, y_train, num_clients=10)

    fl_models = []
    for client in clients:
        model = train_client(client["X"], client["y"])
        fl_models.append(model)

    y_pred_fl = federated_predict(fl_models, X_test)
    acc_fl = accuracy_score(y_test, y_pred_fl)

    print(f"✅ FL Accuracy: {acc_fl*100:.2f}%")

    # ==============================
    # 3. FULL MODEL (FL + SHAP + DP + HADA)
    # ==============================
    print("\n🔴 FL + SHAP + DP (HADA)")

    full_models = federated_training(
        clients,
        rounds=5,
        epsilon=1.0
    )

    y_pred_full = federated_predict(full_models, X_test)
    acc_full = accuracy_score(y_test, y_pred_full)

    print(f"✅ FL + SHAP + DP Accuracy: {acc_full*100:.2f}%")

    # ==============================
    # FINAL RESULTS
    # ==============================
    print("\n📊 FINAL COMPARISON:")
    print(f"Centralized: {acc_central*100:.2f}%")
    print(f"FL: {acc_fl*100:.2f}%")
    print(f"FL + SHAP + DP: {acc_full*100:.2f}%")

    print("\n🎉 ALL DONE")


if __name__ == "__main__":
    main()