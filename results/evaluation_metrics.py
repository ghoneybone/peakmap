import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from scipy.signal import find_peaks
from sklearn.metrics import mean_squared_error, mean_absolute_error

# USER PATHS (modify if needed)
DENOISER_MODEL_PATH = "[path/to/]PeakMap/models/methane_denoiser_model.h5" # INSERT YOUR FILE PATHS HERE (easiest way to find a file path is to drag the data into a terminal window.)
PEAK_MODEL_PATH = "[path/to/]PeakMap/models/methane_peak_model.h5"
DATA_PATH  = "[path/to/]PeakMap/data/methane_dataset_two_column.npz"


print("Loading denoiser model...")
MODEL_PATH = DENOISER_MODEL_PATH
model = load_model(MODEL_PATH)
print("Model loaded successfully.")

print("Loading dataset...")
data = np.load(DATA_PATH)
X_noisy = data["noisy"]
Y_clean = data["clean"]
params  = data["params"]
wavenumber_grid = np.arange(5900, 6200, 0.01)
print(f"Loaded dataset: {X_noisy.shape[0]} spectra, {X_noisy.shape[1]} points each.")

# Add channel dimension
X = np.expand_dims(X_noisy, -1)
Y = np.expand_dims(Y_clean, -1)

# Split same way as training
VAL_SPLIT = int(0.8 * len(X))
X_val, Y_val = X[VAL_SPLIT:], Y[VAL_SPLIT:]
params_val = params[VAL_SPLIT:]

print("Beginning Denoiser Evaluation")

# Run model inference
print("Running inference on validation set...")
Y_pred = model.predict(X_val, verbose=0)

# Flatten for metrics
Y_true_flat = Y_val.reshape(-1)
Y_pred_flat = Y_pred.reshape(-1)

# 1. Reconstruction Error (MSE / MAE)
mse = mean_squared_error(Y_true_flat, Y_pred_flat)
mae = mean_absolute_error(Y_true_flat, Y_pred_flat)

print("\n=== Reconstruction Metrics ===")
print(f"Validation MSE: {mse:.6f}")
print(f"Validation MAE: {mae:.6f}")

# 2. SNR before and after
def compute_snr(signal, baseline):
    noise = np.std(baseline)
    sig = np.max(signal) - np.min(signal)
    return 20 * np.log10(sig / noise)

snr_before = []
snr_after  = []

baseline_mask = (wavenumber_grid > 6150) & (wavenumber_grid < 6200)

for i in range(len(X_val)):
    noisy = X_val[i].squeeze()
    den   = Y_pred[i].squeeze()

    snr_before.append(compute_snr(noisy, noisy[baseline_mask]))
    snr_after.append(compute_snr(den,   den[baseline_mask]))

print("\n=== SNR Metrics ===")
print(f"Average SNR Before Denoising : {np.mean(snr_before):.2f} dB")
print(f"Average SNR After Denoising  : {np.mean(snr_after):.2f} dB")
print(f"Average SNR Improvement       : {np.mean(np.array(snr_after)-np.array(snr_before)):.2f} dB")

# 3. Peak Preservation Metrics
peak_pos_err = []
peak_height_err = []

for i in range(200):  # evaluate subset for speed
    clean = Y_val[i].squeeze()
    den   = Y_pred[i].squeeze()

    clean_peaks, _ = find_peaks(clean, prominence=5)
    den_peaks,   _ = find_peaks(den,   prominence=5)

    if len(clean_peaks) == 0 or len(den_peaks) == 0:
        continue

    for p in clean_peaks:
        idx = np.argmin(np.abs(den_peaks - p))
        closest = den_peaks[idx]
        
        peak_pos_err.append(abs(p - closest) * 0.01)  # convert to cm^-1
        peak_height_err.append(abs(clean[p] - den[closest]))

print("\n=== Peak Preservation Metrics ===")
print(f"Mean peak position error: {np.mean(peak_pos_err):.4f} cm^-1")
print(f"Mean peak height error  : {np.mean(peak_height_err):.4f}")

# 4. Temperature / Pressure Dependence
temp_errors = {}
press_errors = {}

for i in range(len(X_val)):
    t = params_val[i][0]
    p = params_val[i][1]
    err = mean_squared_error(Y_val[i].squeeze(), Y_pred[i].squeeze())

    temp_errors.setdefault(t, []).append(err)
    press_errors.setdefault(p, []).append(err)

# 5. Residual Plot (one example)
idx = 10
clean = Y_val[idx].squeeze()
noisy = X_val[idx].squeeze()
den   = Y_pred[idx].squeeze()
resid = clean - den

plt.figure(figsize=(12,4))
plt.plot(wavenumber_grid, resid, color="purple")
plt.title("Residuals: Clean - Denoised")
plt.xlabel("Wavenumber (cm$^{-1}$)")
plt.ylabel("Residual")
plt.grid(True)
plt.show()

# 6. Example Denoising Plot
plt.figure(figsize=(12,5))
plt.plot(wavenumber_grid, noisy, label="Noisy", alpha=0.5, color='orange')
plt.plot(wavenumber_grid, clean, label="Ground Truth", linewidth=2)
plt.plot(wavenumber_grid, den, label="Denoised", linestyle='--', linewidth=2)
plt.title("Example Denoising Result")
plt.xlabel("Wavenumber (cm$^{-1}$)")
plt.ylabel("Absorption")
plt.legend()
plt.grid(True)
plt.show()

print("\nDenoiser evaluation complete.")





#
# Begin peak-probability model evaluation
#

MODEL_PATH = PEAK_MODEL_PATH
# EVALUATION HYPERPARAMETERS

# How to define "true" peaks from the clean spectrum
TRUE_PEAK_PROMINENCE = 2.0      # in same units as clean spectrum
TRUE_PEAK_HEIGHT     = 35      # set >0 if you want to ignore very small peaks
TRUE_PEAK_MIN_DIST   = 1       # in index units (points)

# How wide to make probability targets (in index units)
TARGET_SIGMA_PTS     = 1.0      # ~2.35-point FWHM; make smaller for sharper

# How to pick peaks from the model's predicted probability curve
PRED_PROB_THRESH     = 0.35     # min height in predicted prob
PRED_MIN_DIST        = 20       # keep predictions from clustering

# How close (in cm^-1) a predicted peak must be to a true peak to count as TP
MATCH_TOL_CM1        = 0.15

# Example index for plots
EXAMPLE_IDX          = 12

# Helper functions
def build_prob_target_from_indices(indices, length, sigma_pts=1.0):
    """
    Given an array of peak indices, build a smooth probability-like target
    by centering narrow Gaussians at each index and taking the max.
    """
    if len(indices) == 0:
        return np.zeros(length, dtype=float)

    x = np.arange(length)[None, :]          # shape (1, L)
    centers = indices[:, None]              # shape (P, 1)

    gaussians = np.exp(-0.5 * ((x - centers) / sigma_pts) ** 2)
    target = gaussians.max(axis=0)          # (L,) use max to avoid over-summing

    max_val = target.max()
    if max_val > 0:
        target = target / max_val           # normalize to [0, 1]

    return target


def find_true_peaks(clean_spectrum):
    """
    Define ground-truth peaks directly from the clean spectrum
    using SciPy's find_peaks and user-chosen thresholds.
    """
    peaks, props = find_peaks(
        clean_spectrum,
        prominence=TRUE_PEAK_PROMINENCE,
        height=TRUE_PEAK_HEIGHT,
        distance=TRUE_PEAK_MIN_DIST,
    )
    return peaks


# Load model and data
print("Loading model...")
peak_model = load_model(MODEL_PATH)
print("Peak detector model loaded successfully.")

print("Loading dataset...")
data = np.load(DATA_PATH)
X_noisy = data["noisy"]
Y_clean = data["clean"]
params  = data["params"]

wavenumber = np.arange(5900, 6200, 0.01)
dx = wavenumber[1] - wavenumber[0]      # 0.01 cm^-1 spacing

# Add channel dimension for model input
X_clean = np.expand_dims(Y_clean, -1)

# Train/val split (same as training)
VAL_SPLIT = int(0.8 * len(X_clean))
X_val_clean = X_clean[VAL_SPLIT:]
Y_val_clean = Y_clean[VAL_SPLIT:]
params_val = params[VAL_SPLIT:]

N_val, L, _ = X_val_clean.shape
print(f"Validation set: {N_val} spectra, {L} points each")

# Build ground-truth targets and true-peak indices
print("Generating ground-truth peak probability targets...")
Y_val_targets = []
true_peak_indices = []   # list of 1D index arrays

for i in range(N_val):
    clean_spec = Y_val_clean[i]  # shape (L,)

    # 1) "True" peaks directly from clean spectrum
    peaks = find_true_peaks(clean_spec)
    true_peak_indices.append(peaks)

    # 2) Probability-like target from these peak indices
    target_curve = build_prob_target_from_indices(
        peaks,
        length=L,
        sigma_pts=TARGET_SIGMA_PTS,
    )
    Y_val_targets.append(target_curve)

Y_val_targets = np.array(Y_val_targets)[..., None]   # (N_val, L, 1)

print("Targets built.")
print("  Y_val_targets shape:", Y_val_targets.shape)

# Run model inference on validation set
print("Running peak model predictions...")
Y_pred_prob = peak_model.predict(X_val_clean, verbose=0)  # (N_val, L, 1)

# 1. Probability-Level Metrics
mse = mean_squared_error(Y_val_targets.reshape(-1), Y_pred_prob.reshape(-1))
mae = mean_absolute_error(Y_val_targets.reshape(-1), Y_pred_prob.reshape(-1))

print("\n=== Probability-Level Metrics ===")
print(f"MSE (probability curve): {mse:.6f}")
print(f"MAE (probability curve): {mae:.6f}")

# 2. Peak Detection Metrics (precision / recall / F1)
TP_total = 0
FP_total = 0
FN_total = 0

pos_errors = []  # peak position errors in cm^-1

for i in range(N_val):
    target = Y_val_targets[i].squeeze()     # (L,)
    pred   = Y_pred_prob[i].squeeze()       # (L,)

    # True peak indices from our stored list
    true_peaks = true_peak_indices[i]

    # Predicted peaks from the model’s probability curve
    pred_peaks, _ = find_peaks(
        pred,
        height=PRED_PROB_THRESH,
        distance=PRED_MIN_DIST,
    )

    # Match predicted peaks to true peaks
    matched_true = set()
    matched_pred = set()

    for p_pred in pred_peaks:
        if len(true_peaks) == 0:
            continue

        # nearest true peak index
        idx = np.argmin(np.abs(true_peaks - p_pred))
        p_true = true_peaks[idx]

        # convert index offset to cm^-1
        err_cm1 = abs(p_true - p_pred) * dx

        if err_cm1 < MATCH_TOL_CM1:
            matched_true.add(int(p_true))
            matched_pred.add(int(p_pred))
            pos_errors.append(err_cm1)

    TP = len(matched_true)
    FP = len(pred_peaks) - len(matched_pred)
    FN = len(true_peaks) - len(matched_true)

    TP_total += TP
    FP_total += FP
    FN_total += FN

precision = TP_total / (TP_total + FP_total + 1e-9)
recall    = TP_total / (TP_total + FN_total + 1e-9)
F1        = 2 * precision * recall / (precision + recall + 1e-9)

print("\n=== Peak Detection Metrics ===")
print(f"True Positives:  {TP_total}")
print(f"False Positives: {FP_total}")
print(f"False Negatives: {FN_total}")
print("-----------------------------")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {F1:.4f}")

# 3. Peak Position Error Metrics
print("\n=== Peak Position Error ===")
if len(pos_errors) > 0:
    print(f"Mean peak position error:   {np.mean(pos_errors):.4f} cm^-1")
    print(f"Median peak position error: {np.median(pos_errors):.4f} cm^-1")
else:
    print("No matched peaks")

# 4. Diagnostic Plots
idx = EXAMPLE_IDX % N_val

clean = Y_val_clean[idx].squeeze()
target = Y_val_targets[idx].squeeze()
pred = Y_pred_prob[idx].squeeze()
true_peaks = true_peak_indices[idx]
pred_peaks, _ = find_peaks(pred, height=PRED_PROB_THRESH, distance=PRED_MIN_DIST)

plt.figure(figsize=(14,6))
plt.plot(wavenumber, target, label="Ground Truth Probability", linewidth=2)
plt.plot(wavenumber, pred, label="Predicted Probability", linewidth=2, alpha=0.7)
plt.scatter(wavenumber[true_peaks],
            target[true_peaks],
            color='blue',
            label="True Peaks")
plt.scatter(wavenumber[pred_peaks],
            pred[pred_peaks],
            color='red',
            label="Predicted Peaks")
plt.legend()
plt.title("Example Clean vs Predicted Peak Probability Curve")
plt.xlabel("Wavenumber (cm$^{-1}$)")
plt.ylabel("Probability")
plt.grid(True)
plt.show()

# Histogram of peak localization errors
plt.figure(figsize=(10,5))
plt.hist(pos_errors, bins=40)
plt.title("Distribution of Peak Position Errors")
plt.xlabel("Error (cm$^{-1}$)")
plt.ylabel("Count")
plt.grid(True)
plt.show()

print("\nPeak detector evaluation complete.")
