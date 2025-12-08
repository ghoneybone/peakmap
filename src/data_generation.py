# IMPORTANT: This file takes hours to converge. Run on HPC cluster or external server if possible.

from hapi import *
import h5py
import matplotlib.pyplot as plt
import numpy as np

# Make a local database
hapi.db_begin("data")

# Fetch a specific range of methane
hapi.fetch('CH4', 6, 1, 5900, 6200)

# Download 5000 pairs of data

# Define common wavenumber grid
wavenumber_grid = np.arange(5900,6200,0.01)

# Define Lists of Clean Spectrums, Noisy Spectrums, and Parameters
clean_spectrums = []
noisy_spectrums = []
parameters = []

# Initialize Parameters
NUM_SAMPLES = 5000 # 5000 for training CHANGE THIS FOR SHORTER RUNTIMES
SNR = 30
PATH_LENGTH = 1000.0

def generate_spectrum(params):
    """
    Generates one clean/noisy spectrum pair.
    'params' is a tuple: (T_rand, P_rand)
    """
    T_rand, P_rand = params
    
    # 1. Calculate the clean, continuous spectrum
    # The correct function call for HAPI
    nu_clean, S_clean = hapi.absorptionCoefficient_Voigt(
        SourceTables='CH4',           # Name of the table (string)
        OmegaStep=0.01,               # Wavenumber step
        HITRAN_units=False,           # Use cm^-1 for absorption coefficient
        OmegaWing=50,                 # Wing cutoff
        Environment={'T': T_rand, 'p': P_rand},  # Temperature and pressure
        GammaL='gamma_self'           # Lorentzian broadening
    )
    
    # Interpolate grid
    S_clean_interp = np.interp(wavenumber_grid, nu_clean, S_clean)
    S_clean = S_clean_interp * PATH_LENGTH # Apply path length to get absorbance

        # Add realistic noise
    S_noisy = add_realistic_noise(
        S_clean, 
        snr=SNR,
        baseline_drift=True,
        shot_noise=True,
        spike_probability=0.0005  # ~0.05% of points are spikes
    )
    
    # 5. Return the results
    return (S_clean, S_noisy)

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

# Data Download Loop
for _ in range(NUM_SAMPLES):
    
    # Randomize Temp (280K -> 320K) [Float]
    rand_temp = np.random.uniform(280,320)
    # Randomize Pressure (0.8 Atm -> 1.2 Atm) [Float]
    rand_pressure = np.random.uniform(0.8,1.2)

    params_tuple = (rand_temp, rand_pressure)
    S_clean, S_noisy = generate_spectrum(params_tuple)

    # Store results
    clean_spectrums.append(S_clean)
    noisy_spectrums.append(S_noisy)
    parameters.append([rand_temp, rand_pressure])
    print(f"Round {_ + 1} / {NUM_SAMPLES} downloaded: {(_ + 1)/NUM_SAMPLES}% complete")

# Convert to Numpy array
clean_array = np.array(clean_spectrums)
noisy_array = np.array(noisy_spectrums)
params_array = np.array(parameters)

# Save using HDF5
print("Saving data to methane_dataset.npz")
np.savez_compressed(
    'methane_dataset.npz',
    noisy=noisy_array,
    clean=clean_array,
    params=params_array
)
print("...Save complete.")