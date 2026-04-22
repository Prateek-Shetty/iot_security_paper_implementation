import matplotlib.pyplot as plt

# ==============================
# CLIENT COUNTS
# ==============================
clients = [5, 25, 50, 75, 100]

# ==============================
# YOUR DATA (PUT YOUR FINAL VALUES HERE)
# ==============================

centralized = [87.82, 87.82, 87.82, 87.82, 87.82]

fl = [88.93, 91.47, 91.93, 92.44, 93.58]

shap_dp = [89.01, 91.33, 91.99, 92.66, 93.37]

energy = [88.95, 88.75, 89.02, 88.72, 88.99]

# ==============================
# PLOT
# ==============================
plt.figure(figsize=(10, 6))

plt.plot(clients, centralized, marker='o', label='Centralized')
plt.plot(clients, fl, marker='o', label='FL')
plt.plot(clients, shap_dp, marker='o', label='FL+SHAP+DP')
plt.plot(clients, energy, marker='o', label='Energy-Aware FL')

plt.xlabel("Number of Clients")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs Number of Clients")
plt.legend()
plt.grid()

# ==============================
# SAVE GRAPH
# ==============================
plt.savefig("results/graphs/clients_vs_accuracy.png")

print("✅ Graph saved as clients_vs_accuracy.png")

plt.show()