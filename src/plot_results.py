import matplotlib.pyplot as plt

# ==============================
# YOUR FINAL RESULTS
# ==============================
models = ["Centralized", "FL", "FL+SHAP+DP"]
accuracy = [87.86, 87.61, 87.32]
f1_scores = [0.9240, 0.9223, 0.9203]

# ==============================
# ACCURACY BAR GRAPH
# ==============================
plt.figure()

plt.bar(models, accuracy)
plt.title("Model Comparison (Accuracy)")
plt.xlabel("Models")
plt.ylabel("Accuracy (%)")

for i, v in enumerate(accuracy):
    plt.text(i, v + 0.2, f"{v:.2f}", ha='center')

plt.show()

# ==============================
# F1 SCORE GRAPH
# ==============================
plt.figure()

plt.bar(models, f1_scores)
plt.title("Model Comparison (F1 Score)")
plt.xlabel("Models")
plt.ylabel("F1 Score")

for i, v in enumerate(f1_scores):
    plt.text(i, v + 0.001, f"{v:.4f}", ha='center')

plt.show()