import numpy as np


# ==============================
# GRADIENT CLIPPING (PAPER)
# ==============================
def clip_weights(weights, clip_value=1.0):
    """
    Clip weights to limit sensitivity

    Paper concept:
    → prevents large updates (privacy leakage)

    Returns:
    - clipped weights
    """

    clipped = []

    for w in weights:
        clipped.append(np.clip(w, -clip_value, clip_value))

    return clipped


# ==============================
# ADD DIFFERENTIAL PRIVACY NOISE
# ==============================
def add_dp_noise(weights, epsilon=1.0, delta=1e-5):
    """
    Apply Gaussian noise for Differential Privacy

    Parameters:
    - epsilon: privacy budget (lower = more privacy)
    - delta: small constant

    Returns:
    - noisy weights
    """

    # standard deviation based on epsilon
    sigma = 2.0 / epsilon   # 🔥 stronger noise (realistic FL)

    noisy_weights = []

    for w in weights:
        noise = np.random.normal(0, sigma, size=w.shape)
        noisy_weights.append(w + noise)

    return noisy_weights


# ==============================
# FULL DP PIPELINE
# ==============================
def apply_differential_privacy(weights, epsilon=1.0):
    """
    Full DP pipeline:
    1. Clip weights
    2. Add noise

    Returns:
    - DP-protected weights
    """

    clipped = clip_weights(weights)
    noisy = add_dp_noise(clipped, epsilon)

    return noisy