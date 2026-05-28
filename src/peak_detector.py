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
plt.rcParams['figure.figsize'] = (12, 5)

# Load Data
data = np.load('methane_dataset.npz', allow_pickle=True)
denoiser_model = tf.keras.models.load_model('methane_denoiser.h5') 

X_noisy = data['noisy']          # (N, WINDOW_POINTS)
Y_clean = data['clean']          # (N, WINDOW_POINTS)
params  = data['params']         # (N, 5) → [spec_idx, T, P, scale, center]

COL_SPEC_IDX = 0
COL_T        = 1
COL_P        = 2
COL_SCALE    = 3
COL_CENTER   = 4  

DX           = 0.01                        # cm^-1 per point
WINDOW_POINTS = Y_clean.shape[1]           # e.g. 2048
HALF_WIDTH    = (WINDOW_POINTS // 2) * DX  # half-width in cm^-1

print(f"Dataset : {X_noisy.shape[0]} windows, {WINDOW_POINTS} points/window")
print(f"Half-width : {HALF_WIDTH:.2f} cm^-1")
print(f"params columns : spec_idx | T | P | scale | center")


def window_wavenumber(sample_idx):
    """Return the wavenumber array for window sample_idx."""
    center = params[sample_idx, COL_CENTER]
    return np.linspace(center - HALF_WIDTH, center + HALF_WIDTH, WINDOW_POINTS)


# Peak Probability Target Generation

# Load HITRAN-grounded peak offsets saved during data generation
peak_offsets = data['peak_offsets']  # object array, variable length per chunk

def positions_to_probability_target(offsets, n_points=WINDOW_POINTS,
                                     window_width=20.48, sigma_cm=0.02):
    """
    Convert HITRAN peak offsets (cm^-1 relative to chunk center) to a
    smooth probability target. sigma_cm should match your Voigt HWHM (~0.05 cm^-1).
    """
    target = np.zeros(n_points, dtype=np.float32)
    if len(offsets) == 0:
        return target

    x_cm = np.linspace(-window_width/2, window_width/2, n_points)
    for offset in offsets:
        target += np.exp(-0.5 * ((x_cm - offset) / sigma_cm) ** 2)

    return np.clip(target / (target.max() + 1e-8), 0, 1)

print("Generating HITRAN-grounded peak probability targets...")
Y_peak_targets = np.zeros((len(Y_clean), WINDOW_POINTS), dtype=np.float32)
for i in range(len(Y_clean)):
    if i % 5000 == 0:
        print(f"  {i}/{len(Y_clean)}")
    Y_peak_targets[i] = positions_to_probability_target(peak_offsets[i])


# Add channel dimension: (N, WINDOW_POINTS, 1)
print("Generating denoised training inputs...")
X_denoised = np.zeros_like(X_noisy, dtype=np.float32)
batch_size = 256  # adjust to fit your GPU memory
for start in range(0, len(X_noisy), batch_size):
    end = min(start + batch_size, len(X_noisy))
    batch = X_noisy[start:end, :, np.newaxis].astype(np.float32)
    X_denoised[start:end] = denoiser_model.predict(batch, verbose=0).squeeze()
    if start % 10000 == 0:
        print(f"  {start}/{len(X_noisy)}")

X_peak_input = np.expand_dims(X_denoised, axis=-1).astype(np.float32)
Y_peak_targets = np.expand_dims(Y_peak_targets, axis=-1).astype(np.float32)

print(f"X_peak_input   shape : {X_peak_input.shape}")
print(f"Y_peak_targets shape : {Y_peak_targets.shape}")
assert X_peak_input.shape == Y_peak_targets.shape, \
    f"Shape mismatch: {X_peak_input.shape} vs {Y_peak_targets.shape}"
assert X_peak_input.shape[-1] == 1, "Missing channel dimension"
print("Shape check passed – ready for training.")


# Target Sanity Plot
idx = 1
wn_idx = window_wavenumber(idx)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(wn_idx, Y_clean[idx], label='Clean spectrum')
plt.title(f'Clean spectrum #{idx}')
plt.xlabel('Wavenumber (cm⁻¹)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(wn_idx, Y_peak_targets[idx].squeeze(), color='purple',
         label='Peak probability target')
plt.title('Ground-truth peak probability density')
plt.xlabel('Wavenumber (cm⁻¹)')
plt.legend()

plt.tight_layout()
plt.show()


# Model Definition

def build_peak_detector(n_points, filters=32, kernel_size=7, n_blocks=4):
    """
    Lightweight 1-D CNN that maps a denoised spectrum → peak probability density.
    Architecture: stacked dilated conv blocks with a sigmoid output.
    """
    inputs = layers.Input(shape=(n_points, 1))
    x = inputs

    # Encoder: dilated convolutions to capture features at multiple scales
    for i in range(n_blocks):
        dilation = 2 ** i          # 1, 2, 4, 8
        x = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding='causal',
            dilation_rate=dilation,
            activation='relu'
        )(x)
        x = layers.BatchNormalization()(x)

    # Optional wider context layer
    x = layers.Conv1D(filters * 2, kernel_size=1, activation='relu')(x)
    x = layers.Dropout(0.1)(x)

    # Output: one probability value per point, bounded [0, 1]
    outputs = layers.Conv1D(1, kernel_size=1, activation='sigmoid')(x)

    return models.Model(inputs, outputs, name='peak_detector')

peak_model = build_peak_detector(WINDOW_POINTS)

# Weighted MSE: penalise errors near peaks much more than baseline errors
def weighted_mse(y_true, y_pred):
    weight = 1.0 + 9.0 * y_true   # peak regions weighted 10x over baseline
    return tf.reduce_mean(weight * tf.square(y_true - y_pred))

peak_model.compile(optimizer='adam', loss=weighted_mse, metrics=['mae'])
peak_model.summary()


# Training
es  = callbacks.EarlyStopping(patience=15, restore_best_weights=True, verbose=1)
rlp = callbacks.ReduceLROnPlateau(patience=8, factor=0.5, verbose=1)

history = peak_model.fit(
    X_peak_input, Y_peak_targets,
    validation_split=0.2,
    epochs=150,
    batch_size=32,
    callbacks=[es, rlp],
    verbose=1
)

peak_model.save('peak_probability_model.h5')
print("Peak model saved →", os.path.abspath('peak_probability_model.h5'))