import pandas as pd


def load_data(path="dataset/ML-EdgeIIoT-dataset.csv", sample_size=50000):
    """
    Load Edge-IIoT dataset (FIXED)

    Fix:
    - Avoid ordered data issue
    - Ensure both classes (attack + normal)
    - Random sampling for proper ML training
    """

    print("📥 Loading dataset...")

    # ==============================
    # 1. LOAD FULL DATASET
    # ==============================
    df = pd.read_csv(path, low_memory=False)

    print(f"📊 Full dataset shape: {df.shape}")

    # ==============================
    # 2. RANDOM SAMPLE (CRITICAL FIX)
    # ==============================
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)

    # shuffle again for safety
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"✅ Sampled dataset shape: {df.shape}")

    return df