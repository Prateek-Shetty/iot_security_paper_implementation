import numpy as np
from lightgbm import LGBMClassifier


# ==============================
# CREATE CLIENTS (NON-IID FIXED)
# ==============================
def create_clients(X, y, num_clients=5, shuffle=True):
    """
    Balanced NON-IID client split (FINAL FIX)
    """

    print("📡 Creating clients...")

    # convert y to numpy
    if hasattr(y, "values"):
        y_np = y.values
    else:
        y_np = y

    # split by class
    idx_class0 = np.where(y_np == 0)[0]
    idx_class1 = np.where(y_np == 1)[0]

    np.random.shuffle(idx_class0)
    np.random.shuffle(idx_class1)

    clients = []

    # split each class across clients
    c0_chunks = np.array_split(idx_class0, num_clients)
    c1_chunks = np.array_split(idx_class1, num_clients)

    for i in range(num_clients):
        # 🔥 mix both classes (IMPORTANT)
        indices = np.concatenate([c0_chunks[i], c1_chunks[i]])
        np.random.shuffle(indices)

        clients.append({
            "id": i,
            "X": X[indices],
            "y": y_np[indices]
        })

    print(f"✅ {len(clients)} clients created (BALANCED NON-IID)")

    return clients

# ==============================
# LOCAL TRAINING (CLEANED)
# ==============================
def train_client(X, y):

    model = LGBMClassifier(
        n_estimators=40,
        max_depth=5,
        learning_rate=0.1,
        class_weight="balanced",
        verbosity=-1
    )

    model.fit(X, y)

    return model


# ==============================
# EXTRACT MODEL WEIGHTS
# ==============================
def get_model_weights(model):
    return model.booster_.feature_importance()