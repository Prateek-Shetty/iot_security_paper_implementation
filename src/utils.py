import numpy as np
from sklearn.metrics import accuracy_score


# ==============================
# EVALUATE MODEL
# ==============================
def evaluate_model(model, X_test, y_test):
    """
    Evaluate model accuracy
    """

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"📊 Accuracy: {acc * 100:.2f}%")

    return acc


# ==============================
# SPLIT TRAIN / TEST
# ==============================
def train_test_split_data(X, y, test_size=0.2):
    from sklearn.model_selection import train_test_split

    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )


# ==============================
# SIMPLE LOGGING
# ==============================
def log(message):
    print(f"[LOG] {message}")