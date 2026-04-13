from src.data_loader import load_data
from src.preprocessing import preprocess
from src.client import create_clients, train_client

# Load data
df = load_data()

# Preprocess
X, y = preprocess(df)

# Create clients
clients = create_clients(X, y, num_clients=10)

print("Clients created:", len(clients))

# Train locally
models = []

for X_c, y_c in clients:
    model = train_client(X_c, y_c)
    models.append(model)

print("Local training complete")