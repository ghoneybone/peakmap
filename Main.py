import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model
from scipy.signal import find_peaks

# Load trained models
denoiser = load_model("[path/to/]models/methane_denoiser_model.h5")
peak_model = load_model("[path/to/]models/methane_peak_model.h5", compile=False)

print("Models loaded successfully.\n")

df = pd.read_csv("[path/to/]data/test_spectrum.csv")

wavenumber = df.iloc[:, 0].values
noisy = df.iloc[:, 1].values

# Load test spectrum
print(f"Loaded test spectrum: {len(noisy)} points\n")

# Run Inference
def run_models(noisy_spectrum, denoiser, peak_model,
               wn, prob_threshold=0.35, min_distance_points=20):
    # Denoise
    denoised = denoiser.predict(noisy_spectrum[None, ..., None], verbose=0).squeeze()

    # Peak probability
    prob = peak_model.predict(denoised[None, ..., None], verbose=0).squeeze()

    # Detect peaks in probability curve
    peak_idx, _ = find_peaks(prob, height=prob_threshold, distance=min_distance_points)

    return denoised, prob, peak_idx, prob[peak_idx]

denoised, prob, peak_idx, conf = run_models(
    noisy, denoiser, peak_model,
    wn=wavenumber,
    prob_threshold=0.35
)

peak_wavenumbers = wavenumber[peak_idx]

# 4. Plot results
plt.figure(figsize=(16,10))

# 1. Noisy vs denoised
plt.subplot(3,1,1)
plt.plot(wavenumber, noisy, label="Noisy", alpha=0.6)
plt.plot(wavenumber, denoised, label="Denoised", lw=2)
plt.legend(); plt.title("Stage 1: Denoising")

# 2. Probability map
plt.subplot(3,1,2)
plt.plot(wavenumber, prob, label="Peak Probability", color="purple")
plt.scatter(peak_wavenumbers, conf, color="red")
plt.legend(); plt.title("Stage 2: Peak Probability Map")

# 3. Detected peaks on denoised spectrum
plt.subplot(3,1,3)
plt.plot(wavenumber, denoised, lw=2)
plt.scatter(peak_wavenumbers, denoised[peak_idx], color="red", s=50)
plt.title("Final Detected Peaks")
plt.xlabel("Wavenumber (cm⁻¹)")

plt.tight_layout()
plt.show()

# 5. Print peak wavenumbers + confidence scores
print(f"\nDetected {len(peak_idx)} peaks:\n")
for wn_val, c_val in zip(peak_wavenumbers, conf):
    print(f"  wavenumber = {wn_val:.3f} cm⁻¹, confidence = {c_val:.3f}")
