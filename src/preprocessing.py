import numpy as np

def preprocess(df):
    # shuffle data (important for FL realism)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # binary classification (as per paper)
    y = df["Attack_label"]

    # drop label columns
    df = df.drop(["Attack_label", "Attack_type"], axis=1)

    # keep numeric only
    df = df.select_dtypes(include=[np.number])

    # clean data
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    X = df.values

    return X, y