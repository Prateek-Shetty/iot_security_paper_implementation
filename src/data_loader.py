import pandas as pd

def load_data():
    df = pd.read_csv("dataset/ML-EdgeIIoT-dataset.csv", nrows=50000)
    return df