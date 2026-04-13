import numpy as np

# DIFFERENTIAL PRIVACY (paper method)
def add_noise(weights, sigma=0.1):
    noisy_weights = []

    for w in weights:
        noise = np.random.normal(0, sigma, size=w.shape)
        noisy_weights.append(w + noise)

    return noisy_weights