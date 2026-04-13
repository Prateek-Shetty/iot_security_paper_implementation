import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess(df, mode="binary"):

    print("🧹 Preprocessing started...")

    # ==============================
    # 1. SHUFFLE
    # ==============================
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # ==============================
    # 2. LABEL
    # ==============================
    if mode == "binary":
        y = df["Attack_label"]
        print("🎯 Mode: Binary Classification (Attack_label)")
    else:
        y = df["Attack_type"]
        print("🎯 Mode: Multiclass (Attack_type)")

    # 🔍 DEBUG LABEL DISTRIBUTION
    print("📊 Label Distribution BEFORE balancing:")
    print(y.value_counts())

    # ==============================
    # 3. SAFE BALANCING (FIXED)
    # ==============================
    if mode == "binary":
        df["Attack_label"] = y

        classes = df["Attack_label"].unique()

        # only balance if both classes exist
        if len(classes) == 2:
            df_major = df[df["Attack_label"] == classes[0]]
            df_minor = df[df["Attack_label"] == classes[1]]

            min_size = min(len(df_major), len(df_minor))

            df_major = df_major.sample(min_size, random_state=42)
            df_minor = df_minor.sample(min_size, random_state=42)

            df = pd.concat([df_major, df_minor]).sample(frac=1).reset_index(drop=True)

            y = df["Attack_label"]

            print(f"⚖ Balanced dataset: {len(df)} samples")

        else:
            print("⚠ Skipping balancing (only one class present)")

    # ==============================
    # 4. DROP LABELS
    # ==============================
    df = df.drop(["Attack_label", "Attack_type"], axis=1)

    # ==============================
    # 5. NUMERIC ONLY
    # ==============================
    df = df.select_dtypes(include=[np.number])

    # ==============================
    # 6. CLEAN
    # ==============================
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    # ==============================
    # 7. REMOVE SPARSE (LESS AGGRESSIVE)
    # ==============================
    threshold = 0.99

    cols_to_drop = []

    for col in df.columns:
        if (df[col] == 0).mean() > threshold:
            cols_to_drop.append(col)

    df = df.drop(columns=cols_to_drop)

    print(f"🗑 Removed sparse columns: {len(cols_to_drop)}")

    # ==============================
    # 8. SCALE
    # ==============================
    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)

    print(f"✅ Final shape: {X.shape}")

    return X, y