# IMPORTANT: This file takes hours to converge. Run on HPC cluster or external server if possible.

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
import warnings
import os
warnings.filterwarnings('ignore')

# %matplotlib inline
plt.rcParams['figure.figsize'] = (12, 5)

# data = np.load('../data/methane_dataset_full.npz')
data = np.load('[path/to/]data/methane_dataset_full.npz') # EDIT FILE PATH HERE


X_noisy = data['noisy']      # shape: (N, 3000)
Y_clean = data['clean']      # shape: (N, 3000)
params = data['params']

wavenumber = np.arange(5900, 6200, 0.01)
print(f"Dataset: {X_noisy.shape[0]} spectra, {wavenumber.shape[0]} points each")

X_peak_input = np.expand_dims(Y_clean, -1)

def create_peak_probability_target(spectrum, wn, sigma_cm1=0.5, prominence=2):
    peaks, _ = find_peaks(spectrum, prominence=prominence, distance=15)
    
    target = np.zeros_like(spectrum, dtype=np.float32)
    
    if len(peaks) == 0:
        return target
    
    dx = wn[1] - wn[0]                      # 0.01 cm⁻¹
    sigma_points = sigma_cm1 / (dx * 2.355)  # FWHM → std dev in points
    
    x_grid = wn[None, :]                     # shape (1, 3000)
    centers = wn[peaks][:, None]             # shape (n_peaks, 1)
    
    # Vectorised Gaussian bumps → much faster and no shape bugs
    gaussians = np.exp(-0.5 * ((x_grid - centers) / sigma_points)**2)
    target = gaussians.sum(axis=0)
    
    if target.max() > 0:
        target = target / target.max()
    
    return target

print("Generating peak probability targets...")
Y_peak_targets = np.zeros((len(Y_clean), len(wavenumber)), dtype=np.float32)

for i in range(len(Y_clean)):
    if i % 1000 == 0:
        print(f"   {i}/{len(Y_clean)}")
    Y_peak_targets[i] = create_peak_probability_target(Y_clean[i], wavenumber)

# Final shape must be (N, 3000, 1)
Y_peak_targets = Y_peak_targets[..., np.newaxis]   # adds the channel dimension

print("Y_peak_targets shape :", Y_peak_targets.shape)   # ← should say (N, 3000, 1)
print("X_peak_input   shape :", X_peak_input.shape)         # ← should also be (N, 3000, 1)

idx = 1  # try a few different ones

plt.figure(figsize=(14, 6))
plt.subplot(1,2,1)
plt.plot(wavenumber, Y_clean[idx], label='Clean Spectrum')
plt.title(f'Clean Spectrum #{idx}')
plt.xlabel('Wavenumber (cm⁻¹)')
plt.legend()

plt.subplot(1,2,2)
plt.plot(wavenumber, Y_peak_targets[idx].squeeze(), label='Peak Probability Target', color='purple')
plt.title('Ground-Truth Peak Probability Density')
plt.xlabel('Wavenumber (cm⁻¹)')
plt.legend()
plt.tight_layout()
plt.show()

# 5. Build the Peak Probability CNN – FIXED FOR YOUR REAL DATA SIZE
SPECTRUM_LENGTH = Y_clean.shape[1]   # ← automatically correct (30000 in your case)

def build_peak_detector():
    inputs = layers.Input(shape=(SPECTRUM_LENGTH, 1))
    
    x1 = layers.Conv1D(32, 7, padding='same', activation='relu')(inputs)
    x1 = layers.BatchNormalization()(x1)
    
    x2 = layers.Conv1D(64, 5, padding='same', activation='relu')(x1)
    x2 = layers.BatchNormalization()(x2)
    
    x3 = layers.Conv1D(96, 3, padding='same', activation='relu')(x2)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Dropout(0.3)(x3)
    
    # Residual
    shortcut = layers.Conv1D(96, 1, padding='same')(inputs)
    x3 = layers.Add()([x3, shortcut])
    
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(x3)
    x = layers.Conv1D(32, 3, padding='same', activation='relu')(x)
    outputs = layers.Conv1D(1, 3, padding='same', activation='sigmoid')(x)
    
    return models.Model(inputs, outputs)

# Re-build the model with correct size
peak_model = build_peak_detector()
peak_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
peak_model.summary()

# FINAL DATA PREPARATION – 100% GUARANTEED TO WORK
X_peak_input = np.expand_dims(Y_clean, axis=-1).astype(np.float32)   # (N, 3000, 1)

# Make sure Y_peak_targets has exactly the same shape
# (the vectorised version already gave us (N, 3000) → we just add the channel)
if Y_peak_targets.ndim == 2:
    Y_peak_targets = Y_peak_targets[..., np.newaxis]   # add channel dimension
Y_peak_targets = Y_peak_targets.astype(np.float32)

print("Final check before training:")
print(f"   X_peak_input.shape   = {X_peak_input.shape}")    # e.g. (10000, 3000, 1)
print(f"   Y_peak_targets.shape = {Y_peak_targets.shape}")  # must be identical
print(f"   X  range : {X_peak_input.min():.4f} … {X_peak_input.max():.4f}")
print(f"   Y  range : {Y_peak_targets.min():.4f} … {Y_peak_targets.max():.4f}")

# Proper assertions
assert X_peak_input.shape == Y_peak_targets.shape, "Shapes do not match!"
assert X_peak_input.shape[-1] == 1, "Missing channel dimension!"

print("All good – ready for training!")

es  = callbacks.EarlyStopping(patience=15, restore_best_weights=True, verbose=1)
rlp = callbacks.ReduceLROnPlateau(patience=8, factor=0.5, verbose=1)

history = peak_model.fit(
    X_peak_input, Y_peak_targets,
    validation_split=0.2,
    epochs=150,
    batch_size=8,
    callbacks=[es, rlp],
    verbose=1
)

peak_model.save('peak_probability_model.h5')
print("Peak model saved as peak_probability_model.h5")
print("Current working directory:", os.getcwd())
print("Full path to model:", os.path.abspath('peak_probability_model.h5'))

def detect_peaks_from_noisy(noisy_spectrum, denoiser, peak_model, wn=wavenumber,
                           prob_threshold=0.4, min_distance_points=20):
    """
    Full two-stage pipeline
    """
    # Stage 1: Denoise
    denoised = denoiser.predict(noisy_spectrum[None, ..., None], verbose=0).squeeze()
    
    # Stage 2: Peak probability
    prob = peak_model.predict(denoised[None, ..., None], verbose=0).squeeze()
    
    # Extract peaks from probability map
    peaks, _ = find_peaks(prob, height=prob_threshold, distance=min_distance_points)
    peak_wn = wn[peaks]
    peak_conf = prob[peaks]
    
    return denoised, prob, peak_wn, peak_conf, peaks

    i = 7  # change this to see different spectra
noisy = X_noisy[i]

denoised, prob, peak_wn, conf, peak_idx = detect_peaks_from_noisy(
    noisy, denoiser=denoiser_model, peak_model=peak_model, prob_threshold=0.35
)

# Safety check
peak_idx = peak_idx.astype(int)

plt.figure(figsize=(16, 9))

# 1. Denoising result
plt.subplot(3,1,1)
plt.plot(wavenumber, noisy, label='Noisy Input', alpha=0.7, color='orange')
plt.plot(wavenumber, denoised, label='Denoised (U-Net)', lw=2, color='blue')
plt.ylabel('Absorption')
plt.title('Stage 1 – Denoising U-Net')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Learned peak probability
plt.subplot(3,1,2)
plt.plot(wavenumber, prob, label='Peak Probability Density', color='purple', lw=2)
plt.plot(peak_wn, conf, 'r^', ms=10)
plt.ylabel('Probability')
plt.title('Stage 2 – Learned Peak Probability (CNN)')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Final peak detection
plt.subplot(3,1,3)
plt.plot(wavenumber, denoised, label='Denoised Spectrum', lw=2, color='black')
plt.plot(peak_wn, denoised[peak_idx], 'r^', ms=14, mec='darkred', mew=2,
         label=f'Detected Peaks: {len(peak_wn)}')

# Add confidence labels above each peak
for wn, c in zip(peak_wn, conf):
    idx = np.argmin(np.abs(wavenumber - wn))           # safest way
    y_pos = denoised[idx] + 0.05 * denoised.max()      # slightly above peak
    plt.text(wn, y_pos, f'{c:.2f}', fontsize=11, ha='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

plt.xlabel('Wavenumber (cm⁻¹)')
plt.ylabel('Absorption')
plt.title('Final Result – Automatic Methane Peak Detection')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()