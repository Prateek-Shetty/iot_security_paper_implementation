# 🔐 Federated Learning with SHAP & Differential Privacy (HADA)

## 📌 Overview

This project implements a **privacy-aware Federated Learning (FL) system** for IoT intrusion detection based on the Edge-IIoT dataset.

The system enhances traditional FL by integrating:

* **SHAP (SHapley Additive Explanations)** → to measure client reliability
* **Differential Privacy (DP)** → to protect client data
* **HADA (Hybrid Adaptive Aggregation)** → to weight client contributions intelligently

---

## 🎯 Objective

To improve federated learning by:

* Identifying **reliable clients** using SHAP-based feature stability
* Ensuring **data privacy** using differential privacy
* Performing **adaptive aggregation** instead of simple averaging

---

## 🧠 Key Idea

Traditional Federated Learning treats all clients equally.

This project introduces:

```
Client Weight ∝ SHAP Stability / Privacy Noise
```

→ More reliable and less noisy clients contribute more to the global model.

---

## 🏗️ Project Structure

```
iot_security_paper_implementation/
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── client.py
│   ├── shap_selection.py
│   ├── dp.py
│   ├── server.py
│   ├── federated.py
│   ├── utils.py
│
├── main.py
├── plot_results.py
└── dataset/
```

---

## ⚙️ Workflow

```
Dataset → Preprocessing → Client Split

Centralized Model
        ↓
Basic Federated Learning
        ↓
FL + SHAP + DP (HADA)
        ↓
Evaluation & Comparison
```

---

## 📂 File-wise Explanation

### 1️⃣ data_loader.py

* Loads the Edge-IIoT dataset
* Shuffles data for randomness
  ✔ Represents **global IoT data source**

---

### 2️⃣ preprocessing.py

* Cleans data (NaN, inf)
* Removes sparse features
* Selects numeric columns
* Scales features
  ✔ Prepares data for ML model

---

### 3️⃣ client.py

* Splits dataset into multiple clients (non-IID)
* Trains LightGBM model per client
* Extracts feature importance (proxy for weights)
  ✔ Simulates **distributed IoT devices**

---

### 4️⃣ shap_selection.py

* Computes SHAP values
* Converts them into a stability score
  ✔ Measures **client reliability**

---

### 5️⃣ dp.py

* Clips model updates
* Adds Gaussian noise
  ✔ Ensures **differential privacy**

---

### 6️⃣ server.py

* Computes HADA weights using SHAP + DP
* Aggregates client updates
  ✔ Core logic of **adaptive aggregation**

---

### 7️⃣ federated.py

* Runs full federated learning loop
* Applies SHAP + DP + HADA
* Combines predictions
  ✔ Implements **complete FL pipeline**

---

### 8️⃣ main.py

* Runs experiments:

  * Centralized
  * Federated Learning
  * FL + SHAP + DP
    ✔ Compares model performance

---

### 9️⃣ utils.py

* Evaluation functions
* Metrics (Accuracy, F1 Score)
  ✔ Standardized evaluation

---

### 🔟 plot_results.py

* Generates comparison graphs
  ✔ Visualizes results

---

## 🔬 Algorithms Used

### ✅ LightGBM

* Used for local model training
* Efficient for tabular IoT data

---

### ✅ SHAP

* Explains feature importance
* Used to compute **client stability score**

---

### ✅ Differential Privacy

* Gaussian noise added to updates
* Prevents data leakage

---

### ✅ HADA (Core Contribution)

Adaptive weighting:

```
w_k = exp( τ * s_k / (ε_k + β) )
```

Where:

* `s_k` → SHAP stability
* `ε_k` → privacy budget
* `τ` → scaling factor

---

## 📊 Results

Example output:

```
Centralized  → Acc: 87.86% | F1: 0.9240
FL           → Acc: 87.53% | F1: 0.9218
FL+SHAP+DP   → Acc: 87.59% | F1: 0.9222
```

---

## 📈 Observations

* Centralized model performs best (full data access)
* Federated Learning slightly lower due to distributed training
* HADA improves client weighting using SHAP
* Differential Privacy introduces slight noise but preserves privacy

---

## ⚠️ Limitations

* LightGBM does not support direct weight updates → ensemble approximation used
* SHAP computed on sample data (not full dataset)
* DP applied to feature importance, not gradients

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python main.py
```

---

## 📊 Generate Graphs

```bash
python plot_results.py
```

---

## 🎓 Conclusion

This project demonstrates a **privacy-aware federated learning framework** that:

* Improves robustness using SHAP-based client evaluation
* Preserves privacy with differential privacy
* Enhances aggregation through HADA

---

## 🧠 One-Line Summary

> A federated learning system where client contributions are adaptively weighted using SHAP-based feature stability and differential privacy for secure and intelligent aggregation.

---

## 👨‍💻 Author

Prateek

---
