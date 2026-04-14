import pandas as pd


def load_data(path="dataset/ML-EdgeIIoT-dataset.csv"):
    """
    Load FULL dataset (final version)

    Fix:
    - Uses entire dataset
    - Ensures proper randomness
    """

    print("📥 Loading FULL dataset...")

    df = pd.read_csv(path, low_memory=False)

    print(f"📊 Dataset shape: {df.shape}")

    # 🔥 CRITICAL: shuffle entire dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("✅ Dataset shuffled")

    return df