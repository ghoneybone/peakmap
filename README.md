## PeakMap: Machine-Learning Peak Detection for Methane Spectroscopy

PeakMap is a machine-learning pipeline for identifying absorption peaks in methane spectroscopy data.  
It uses a two-stage architecture:

1. A 1D U-Net denoising autoencoder  
2. A convolutional peak-probability model  

The system takes noisy input spectra and outputs a denoised spectrum, a peak-probability curve, and final peak locations.

---

## Download Required Datasets

PeakMap requires two datasets to run inference.  
Please download them from the links below and place them in the `data/` directory.

### **Dataset 1 — Full Methane Dataset (~3 GB)**  
Direct download:  
https://drive.google.com/uc?export=download&id=15CZs7S9iLqvYrWwyxYPukUPaN1pjvuTb

Save as:
data/methane_dataset_full.npz


---

### **Dataset 2 — Two-Column Methane Dataset**  
Direct download:  
https://drive.google.com/uc?export=download&id=13RLNwhtnbIUki3MW3p2hRq0pfKx-J3E1

Save as:
data/methane_dataset_two_column.npz


---

Once the datasets are in place, you can run the inference scripts normally.
