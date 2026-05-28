# IMPORTANT: This file takes hours to converge. Run on HPC cluster or external server if possible.

from hapi import *
import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv1D, MaxPool1D, UpSampling1D, Concatenate
from tensorflow.keras.models import Model
import tensorflow as tf
import warnings
import os
warnings.filterwarnings('ignore')

# %matplotlib inline
plt.rcParams['figure.figsize'] = (12, 5)

print("Loading data from methane_dataset.npz...")
data = np.load('methane_dataset.npz') # Changed to updated dataset

Y_full = data['clean']  
X_full = data['noisy']
params_array = data['params']

N_POINTS = X_full.shape[1]  # 2048
WINDOW_WIDTH = 20.48
wavenumber_grid = np.linspace(-WINDOW_WIDTH/2, WINDOW_WIDTH/2, N_POINTS)

print(f"Successfully loaded 'noisy' array with shape: {Y_full.shape}")
print(f"Loaded data shapes: X={X_full.shape}, Y={Y_full.shape}")

# Plot one example
sample_index = 0
plt.figure(figsize=(12, 6))
plt.plot(wavenumber_grid, Y_full[sample_index], label='Clean Spectrum')
plt.plot(wavenumber_grid, X_full[sample_index], label='Noisy Spectrum', alpha=0.6)
plt.xlabel('Wavenumber (cm⁻¹)')
plt.ylabel('Absorption (arb. units)')
plt.title('Example CH4 Absorption Spectrum (Continuous, Broadened)')
plt.legend()
plt.grid(True)
plt.show()

# Access the parameters from the loaded array
print(f"Temperature [K] = {params_array[sample_index][0]}")
print(f"Pressure [Atm] = {params_array[sample_index][1]}")

# Add a "channel" dimension
X_data = np.expand_dims(X_full, axis=-1)
Y_data = np.expand_dims(Y_full, axis=-1)
print(f"Reshaped data shapes: X={X_data.shape}, Y={Y_data.shape}")

X_train, X_val, Y_train, Y_val = train_test_split(
    X_data, 
    Y_data, 
    test_size=0.2, 
    random_state=42
)

# Force concrete numpy arrays so Keras always sees a defined shape
X_train = np.array(X_train, dtype=np.float32)
X_val   = np.array(X_val,   dtype=np.float32)
Y_train = np.array(Y_train, dtype=np.float32)
Y_val   = np.array(Y_val,   dtype=np.float32)

print(f"Training shapes:   X={X_train.shape}, Y={Y_train.shape}")

print(f"Training shapes:   X={X_train.shape}, Y={Y_train.shape}")
print(f"Validation shapes: X={X_val.shape}, Y={Y_val.shape}")
input_shape = (X_train.shape[1], 1)

def build_unet(shape): # Denoising U-net autoencoder architecture
    """Builds a 1D U-Net model."""
    inputs = Input(shape=shape)

    # --- Encoder (Contracting Path) ---
    # Downsample 1
    c1 = Conv1D(16, 3, activation='relu', padding='same')(inputs)
    p1 = MaxPool1D(2, padding='same')(c1)
    
    # Downsample 2
    c2 = Conv1D(32, 3, activation='relu', padding='same')(p1)
    p2 = MaxPool1D(2, padding='same')(c2)
    
    # --- Bottleneck ---
    b = Conv1D(64, 3, activation='relu', padding='same')(p2)
    
    # --- Decoder (Expanding Path) ---
    # Upsample 1
    u1 = UpSampling1D(2)(b)
    m1 = Concatenate()([u1, c2]) # <-- Skip connection 1
    c3 = Conv1D(32, 3, activation='relu', padding='same')(m1)
    
    # Upsample 2
    u2 = UpSampling1D(2)(c3)
    m2 = Concatenate()([u2, c1]) # <-- Skip connection 2
    c4 = Conv1D(16, 3, activation='relu', padding='same')(m2)
    
    # --- Output Layer ---
    # Final convolution, output has 1 channel
    # 'linear' activation means it can output any value (not just 0-1)
    outputs = Conv1D(1, 1, activation='linear')(c4)
    
    # Create the model
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# Build model FIRST
denoiser_model = build_unet(input_shape)
denoiser_model.summary()

# Compile model
denoiser_model.compile(optimizer='adam', loss='mean_squared_error')

# Train Model
print("Starting model training...")
history = denoiser_model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=150, # Change this *****
    batch_size=16,
    verbose=1
)
print("...Training complete.")

# Plot Training History
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Training History')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)
plt.show()

denoiser_model.save('methane_denoiser.h5')
print("Denoiser saved as methane_denoiser.h5")
print("Current working directory:", os.getcwd())
print("Full path to model:", os.path.abspath('methane_denoiser.h5'))