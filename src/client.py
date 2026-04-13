import numpy as np
from lightgbm import LGBMClassifier

# CREATE CLIENTS
def create_clients(X, y, num_clients=10):
    clients = []

    data_size = len(X)
    chunk = data_size // num_clients

    for i in range(num_clients):
        start = i * chunk
        end = (i + 1) * chunk if i != num_clients - 1 else data_size

        clients.append({
            "X": X[start:end],
            "y": y[start:end]
        })

    return clients


# LOCAL TRAINING (paper: LightGBM)
def train_client(X, y):
    model = LGBMClassifier(
        n_estimators=100,
        max_depth=8,
        learning_rate=0.1
    )

    model.fit(X, y)
    return model