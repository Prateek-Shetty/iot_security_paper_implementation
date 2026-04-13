import numpy as np

def preprocess(df):
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    y = df["Attack_type"]

    df = df.drop(["Attack_label", "Attack_type"], axis=1)

    df = df.select_dtypes(include=[np.number])

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    X = df.values

    return X, y