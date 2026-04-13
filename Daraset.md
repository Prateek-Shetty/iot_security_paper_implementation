# 📥 Dataset: Edge-IIoTset

## 🔗 Download Link
https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot

---

# 🧠 What is this Dataset?

The **Edge-IIoTset dataset** is a cybersecurity dataset designed for **Intrusion Detection Systems (IDS)** in IoT and Industrial IoT environments.

It contains:
- Realistic **network traffic data**
- Data generated from IoT devices and edge computing environments
- Both:
  - ✅ Normal (benign) traffic
  - ❌ Malicious (attack) traffic

---

# ⚙️ How the Dataset Was Created

This dataset is generated from a **realistic IoT testbed environment**, which includes:

- Smart IoT devices (sensors, controllers)
- Communication protocols (MQTT, TCP, HTTP, etc.)
- Multi-layer architecture:
  - Device Layer
  - Edge Layer
  - Cloud Layer

It simulates how real IoT networks behave in practical scenarios.

---

# 🔥 Types of Attacks in Dataset

The dataset includes multiple cyber attacks such as:

## 1. DoS / DDoS Attacks
- Flooding the network with traffic

## 2. Information Gathering
- Scanning and probing attacks

## 3. Man-in-the-Middle (MITM)
- Intercepting communication

## 4. Injection Attacks
- Injecting malicious data into the system

## 5. Malware Traffic
- Malicious communication patterns

---

# 🧠 What the Model Learns

Each row in the dataset represents a **network traffic record**.

The model learns patterns based on:
- Packet size
- Source and destination
- Protocol behavior
- Time-related features

### Output:
- `0` → Benign (Normal Traffic)
- `1` → Malicious (Attack Traffic)

---

# 🎯 Why This Dataset?

This dataset is chosen because:

- ✔ It is used in the research paper
- ✔ It represents real-world IoT network behavior
- ✔ It contains multiple types of cyber attacks
- ✔ It supports federated learning experiments
- ✔ It is widely used in cybersecurity research

---

# ⚠️ Important Notes

- Dataset size is large (~1GB+)
- Do not load full dataset initially

### Recommended:
```python
df = pd.read_csv("Edge-IIoTset.csv", nrows=50000)