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

        # mix both classes
        indices = np.concatenate([c0_chunks[i], c1_chunks[i]])
        np.random.shuffle(indices)

        # ==============================
        # 🔥 ENERGY CALCULATION (FIXED)
        # ==============================
        data_size = len(indices)
        class_balance = np.mean(y_np[indices])

        energy = 50 + (data_size / len(X)) * 40 + (class_balance * 10)

        # ==============================
        # CLIENT OBJECT
        # ==============================
        clients.append({
            "id": i,
            "X": X[indices],
            "y": y_np[indices],
            "energy": energy
        })

    print(f"✅ {len(clients)} clients created (BALANCED NON-IID)")

    return clients

# ==============================
# DIRICHLET NON-IID SPLIT (PAPER)
# ==============================
def create_clients_dirichlet(X, y, num_clients=5, alpha=0.3):

    print(f"📡 Creating clients using Dirichlet α={alpha}")

    if hasattr(y, "values"):
        y = y.values

    num_classes = len(np.unique(y))
    data_size = len(X)

    # indices per class
    class_indices = [np.where(y == i)[0] for i in range(num_classes)]

    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        np.random.shuffle(class_indices[c])

        proportions = np.random.dirichlet(alpha * np.ones(num_clients))

        proportions = (proportions / proportions.sum()) * len(class_indices[c])
        proportions = proportions.astype(int)

        start = 0
        for i in range(num_clients):
            end = start + proportions[i]
            client_indices[i].extend(class_indices[c][start:end])
            start = end

    clients = []

    for i in range(num_clients):
        idx = np.array(client_indices[i])
        np.random.shuffle(idx)

        clients.append({
            "id": i,
            "X": X[idx],
            "y": y[idx],
            "energy": np.random.uniform(10, 100)
        })

    print(f"✅ {len(clients)} clients created (Dirichlet NON-IID)")

    return clients

# ==============================
# LOCAL TRAINING (CLEANED)
# ==============================
def train_client(X, y):

    model = LGBMClassifier(
        n_estimators=60,    #40
        max_depth=6,        #5
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