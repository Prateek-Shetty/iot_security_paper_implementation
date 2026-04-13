import numpy as np
from lightgbm import LGBMClassifier


# ==============================
# CREATE CLIENTS (SIMULATE IoT DEVICES)
# ==============================
def create_clients(X, y, num_clients=10, shuffle=True):
    """
    Split dataset into multiple clients (IoT devices)

    Parameters:
    - X: features
    - y: labels
    - num_clients: number of devices
    - shuffle: randomize distribution (non-IID simulation)

    Returns:
    - clients: list of dicts {X, y}
    """

    print("📡 Creating clients...")

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

        # last client gets remaining data
        end = (i + 1) * chunk_size if i != num_clients - 1 else data_size

        X_client = X[start:end]
        y_client = y[start:end]

        clients.append({
            "id": i,
            "X": X_client,
            "y": y_client
        })

    print(f"✅ {len(clients)} clients created")

    return clients


# ==============================
# LOCAL TRAINING (LIGHTGBM - PAPER)
# ==============================
def train_client(X, y):
    """
    Train local model on client data

    Returns:
    - trained LightGBM model
    """

    model = LGBMClassifier(
          n_estimators=100,
          max_depth=8,
          learning_rate=0.1,
          verbosity=-1,        
          min_data_in_leaf=20, 
          min_gain_to_split=0.01
    )

    model.fit(X, y)

    return model


# ==============================
# EXTRACT MODEL WEIGHTS
# ==============================
def get_model_weights(model):
    """
    Extract model weights for aggregation

    (Using feature importance as proxy for weights)
    """

    return model.booster_.feature_importance()