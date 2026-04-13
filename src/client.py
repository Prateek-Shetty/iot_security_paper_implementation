import numpy as np
from lightgbm import LGBMClassifier


# ==============================
# CREATE CLIENTS (DATA SPLIT)
# ==============================
def create_clients(X, y, num_clients=10, shuffle=True):
    """
    Split dataset into multiple clients (simulating IoT devices)
    """

    if shuffle:
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        y = y.iloc[indices] if hasattr(y, "iloc") else y[indices]

    clients = []
    data_size = len(X)
    chunk_size = data_size // num_clients

    for i in range(num_clients):
        start = i * chunk_size

        # last client takes remaining data
        end = (i + 1) * chunk_size if i != num_clients - 1 else data_size

        X_client = X[start:end]
        y_client = y[start:end]

        clients.append({
            "X": X_client,
            "y": y_client
        })

    return clients


# ==============================
# TRAIN LOCAL CLIENT MODEL
# ==============================
def train_client(X, y):
    """
    Train a local model on client data
    """

    model = LGBMClassifier(
        n_estimators=100,
        max_depth=8,
        learning_rate=0.1
    )

    model.fit(X, y)

    return model