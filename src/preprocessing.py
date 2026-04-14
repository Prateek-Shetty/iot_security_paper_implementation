import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess(df, mode="binary"):

    print("🧹 Preprocessing started...")

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    if mode == "binary":
        y = df["Attack_label"]
        print("🎯 Mode: Binary Classification (Attack_label)")
    else:
        y = df["Attack_type"]

    print("📊 Label Distribution:")
    print(y.value_counts())

    # drop labels
    df = df.drop(["Attack_label", "Attack_type"], axis=1)

    # numeric only
    df = df.select_dtypes(include=[np.number])

    # clean
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    # remove useless columns
    df = df.loc[:, (df != 0).any(axis=0)]

    # scale
    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)

    # 🔥 CRITICAL FIX: LIMIT FEATURES
    max_features = 10
    X = X[:, :max_features]

    print(f"✅ Final shape after feature limit: {X.shape}")

    return X, y