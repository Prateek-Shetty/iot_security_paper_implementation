from src.data_loader import load_data
from src.preprocessing import preprocess
from src.client import create_clients
from src.federated import federated_training

# LOAD DATA
df = load_data()

# PREPROCESS
X, y = preprocess(df)

# CREATE CLIENTS
clients = create_clients(X, y, num_clients=10)

print("Clients created:", len(clients))

# FEDERATED TRAINING
global_model = federated_training(clients, rounds=5)

print("✅ Federated Learning Completed")