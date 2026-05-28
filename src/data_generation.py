# Pulled from GitHub Data generations file
# IMPORTANT: This file takes hours to converge. Run on HPC cluster or external server if possible.

from hapi import *
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

# Make a local database
hapi.db_begin("data")

# Fetch a specific range of methane
hapi.fetch('CH4', 6, 1, 2900, 3200) 


# # Define common wavenumber grid
# wavenumber_grid = np.arange(5900,6200,0.01)
OMEGA_STEP = 0.01          # cm^-1
WINDOW_WIDTH = 20.48       # cm^-1 (exactly 2048 points)
N_POINTS = int(WINDOW_WIDTH / OMEGA_STEP)  # 2048

# Define Lists of Clean Spectrums, Noisy Spectrums, and Parameters
clean_spectrums = []
noisy_spectrums = []
parameters = []
peak_targets = [] 

# Initialize Parameters
NUM_SAMPLES = 5000 # 5000 for training CHANGE THIS FOR SHORTER RUNTIMES
SNR = 300
PATH_LENGTH = 1000.0

def generate_spectrum(params):
    T_rand, P_rand = params

    nu, S_clean = hapi.absorptionCoefficient_Voigt(
        SourceTables='CH4',
        OmegaStep=0.01,
        HITRAN_units=False,
        OmegaWing=50,
        Environment={'T': T_rand, 'p': P_rand},
        GammaL='gamma_self'
    )

    S_clean = S_clean * PATH_LENGTH

    # Normalize GLOBALLY before chunking
    S_clean_norm, global_scale = normalize_spectrum(S_clean)
    S_noisy = add_realistic_noise(S_clean, snr=SNR, baseline_drift=True,
                                   shot_noise=True, spike_probability=0.0005)
    S_noisy_norm = S_noisy / global_scale  # same scale applied to noisy

    # Slice AFTER global normalization
    clean_windows, centers = slice_and_resample(nu, S_clean_norm, WINDOW_WIDTH, N_POINTS)
    noisy_windows, _ = slice_and_resample(nu, S_noisy_norm, WINDOW_WIDTH, N_POINTS)

    # No per-chunk normalization loop needed anymore
    scales = [global_scale] * len(clean_windows)  # store global scale for reference

    return clean_windows, noisy_windows, scales, centers


def add_realistic_noise(S_clean, snr=30, baseline_drift=True, shot_noise=True, spike_probability=0.001):
    """
    Add multiple realistic noise sources to a spectrum.
    
    Parameters:
    - snr: Signal-to-noise ratio for Gaussian noise
    - baseline_drift: Add slow baseline variations
    - shot_noise: Add Poisson (shot) noise proportional to signal
    - spike_probability: Probability of cosmic ray spikes
    """
    S_noisy = S_clean.copy()
    peak_signal = np.max(S_clean)
    
    # 1. Gaussian white noise (detector electronic noise)
    gaussian_std = peak_signal / snr
    S_noisy += np.random.normal(0, gaussian_std, S_clean.shape)
    
    # 2. Shot noise (Poisson noise - proportional to sqrt of signal)
    if shot_noise and peak_signal > 0:
        # Scale shot noise contribution
        shot_noise_scale = peak_signal / (snr * 2)
        poisson_noise = np.random.poisson(np.abs(S_clean) / shot_noise_scale) * shot_noise_scale - S_clean
        S_noisy += poisson_noise * 0.3  # Reduce contribution
    
    # 3. Baseline drift (low-frequency variation)
    if baseline_drift:
        x = np.linspace(0, 2*np.pi, len(S_clean))
        # Random slow oscillation
        drift = peak_signal * 0.02 * (
            np.random.randn() * np.sin(x * np.random.uniform(0.5, 2)) +
            np.random.randn() * np.cos(x * np.random.uniform(0.5, 2))
        )
        S_noisy += drift
    
    # 4. Random spikes (cosmic rays, electronic glitches)
    if spike_probability > 0:
        n_spikes = np.random.binomial(len(S_clean), spike_probability)
        spike_positions = np.random.choice(len(S_clean), n_spikes, replace=False)
        spike_amplitudes = np.random.uniform(0.5, 3) * peak_signal / snr
        S_noisy[spike_positions] += spike_amplitudes * np.random.choice([-1, 1], n_spikes)
    
    return S_noisy

def slice_and_resample(nu, S, window_width=WINDOW_WIDTH, n_points=N_POINTS):
    windows = []
    centers_out = []

    start = nu.min()
    stop  = nu.max()

    step = window_width / 2   # 50% overlap
    
    centers = np.arange(
        start + window_width / 2,
        stop  - window_width / 2,
        step
    )

    for c in centers:
        mask = (nu >= c - window_width/2) & (nu < c + window_width/2)
        S_win = S[mask]

        # This should now be exact
        if len(S_win) != n_points:
            continue  # safety guard for floating-point edges

        windows.append(S_win)
        centers_out.append(c)   # <-- store the absolute wavenumber

    return windows, centers_out


def normalize_spectrum(S):
    scale = np.max(np.abs(S)) + 1e-8
    return S / scale, scale

def merge_hitran_lines(nu_peaks, sw_peaks, resolution=0.05):
    """
    Merge HITRAN lines within `resolution` cm^-1 of each other into
    single peaks. Returns merged positions (strength-weighted centroid)
    and merged strengths (summed).
    """
    if len(nu_peaks) == 0:
        return np.array([]), np.array([])

    sort_idx = np.argsort(nu_peaks)
    nu_sorted = nu_peaks[sort_idx]
    sw_sorted = sw_peaks[sort_idx]

    merged_nu = []
    merged_sw = []

    current_nu = [nu_sorted[0]]
    current_sw = [sw_sorted[0]]

    for nu, sw in zip(nu_sorted[1:], sw_sorted[1:]):
        if nu - current_nu[-1] < resolution:
            current_nu.append(nu)
            current_sw.append(sw)
        else:
            # Strength-weighted centroid for position
            total_sw = np.sum(current_sw)
            merged_nu.append(np.dot(current_nu, current_sw) / total_sw)
            merged_sw.append(total_sw)
            current_nu = [nu]
            current_sw = [sw]

    # Don't forget the last group
    total_sw = np.sum(current_sw)
    merged_nu.append(np.dot(current_nu, current_sw) / total_sw)
    merged_sw.append(total_sw)

    return np.array(merged_nu), np.array(merged_sw)

STRENGTH_THRESHOLD = 1e-20  # was 1e-20, raising to above noise floor

def get_hitran_peak_positions(T, P, nu_min, nu_max, 
                               strength_threshold=STRENGTH_THRESHOLD):
    nu_lines = np.array(getColumn('CH4', 'nu'))
    sw_lines = np.array(getColumn('CH4', 'sw'))
    window_mask = (nu_lines >= nu_min) & (nu_lines <= nu_max)
    nu_win = nu_lines[window_mask]
    sw_win = sw_lines[window_mask]
    strong_mask = sw_win > strength_threshold
    return nu_win[strong_mask], sw_win[strong_mask]


def label_chunk_from_hitran(chunk_center, chunk_width, T, P,
                             strength_threshold=STRENGTH_THRESHOLD,
                             resolution=0.05):
    nu_min = chunk_center - chunk_width / 2
    nu_max = chunk_center + chunk_width / 2
    nu_peaks, sw_peaks = get_hitran_peak_positions(
        T, P, nu_min, nu_max, strength_threshold
    )
    nu_peaks, sw_peaks = merge_hitran_lines(nu_peaks, sw_peaks, resolution)

    n_peaks      = len(nu_peaks)
    total_strength = np.sum(sw_peaks) if n_peaks > 0 else 0
    max_strength   = np.max(sw_peaks) if n_peaks > 0 else 0

    if n_peaks == 0:
        return 0, np.array([])

    elif total_strength < 3e-19:      # ~3 lines at noise floor
        return 0, np.array([])

    elif max_strength < 1e-18 and n_peaks <= 8:
        return 1, nu_peaks

    elif total_strength < 1e-16 and n_peaks <= 8:
        return 2, nu_peaks

    else:
        return 3, nu_peaks

for i in range(NUM_SAMPLES):
    rand_temp = np.random.uniform(200, 320)
    rand_pressure = 10 ** np.random.uniform(np.log10(0.1), np.log10(1.1))
    clean_wins, noisy_wins, scales, centers = generate_spectrum((rand_temp, rand_pressure))

    chunk_labels = []
    chunk_offsets = []  # raw peak positions relative to chunk center

    for center, c_win in zip(centers, clean_wins):
        label, nu_peaks_abs = label_chunk_from_hitran(
            center, WINDOW_WIDTH, rand_temp, rand_pressure
        )
        chunk_labels.append(label)
        offsets = nu_peaks_abs - center
        chunk_offsets.append(offsets)

    # if not any(l == 3 for l in chunk_labels):
    #     print(f"Spectrum {i+1} skipped — no dense chunks")
    #     continue

    for c_win, n_win, s, center, label, offsets in zip(
            clean_wins, noisy_wins, scales, centers, chunk_labels, chunk_offsets):
        clean_spectrums.append(c_win)
        noisy_spectrums.append(n_win)
        peak_targets.append(offsets)
        parameters.append([i, rand_temp, rand_pressure, s, center, label])

    print(f"Spectrum {i+1}/{NUM_SAMPLES} processed")

# Convert to Numpy arrays
clean_array = np.array(clean_spectrums)
noisy_array = np.array(noisy_spectrums)
params_array = np.array(parameters)

# peak_targets is variable-length per chunk — must use dtype=object
# Do NOT call np.array(peak_targets) without dtype=object
peak_offsets_array = np.array(peak_targets, dtype=object)

np.savez_compressed(
    '4_13_methane_dataset.npz',
    noisy=noisy_array,
    clean=clean_array,
    params=params_array,
    peak_offsets=peak_offsets_array
)
print("...Save complete.")