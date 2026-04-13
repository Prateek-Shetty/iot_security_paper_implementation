from src.client import train_client
from src.server import aggregate_models
from src.dp import add_noise

# FEDERATED TRAINING LOOP
def federated_training(clients, rounds=5):
    global_model = None

    for r in range(rounds):
        print(f"\n🔁 Round {r+1}")

        local_models = []

        for client in clients:
            model = train_client(client["X"], client["y"])
            local_models.append(model)

        # Apply Differential Privacy
        weights = [m.booster_.feature_importance() for m in local_models]
        noisy_weights = add_noise(weights)

        # Aggregation (HADA simplified)
        global_weights = aggregate_models(local_models)

        print("Aggregation done")

        global_model = local_models[0]  # simplified global model

    return global_model