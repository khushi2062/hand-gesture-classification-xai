"""
Phase 2 — Preprocessing
Loads NinaPro DB5 Subject 1 Exercise 1 EMG data.
Applies bandpass filter (20-99 Hz), notch filter (50 Hz),
z-score normalization, and sliding window segmentation.
Output: windows (710, 50, 16), labels (710,), repetitions (710,)
Dataset: NinaPro DB5, Subject 1, Exercise 1
"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from load import emg, labels, repetition, fs
from scipy.signal import butter, sosfiltfilt, iirnotch , filtfilt

selected_gesture=[1,2,6,7]
mask=np.isin(labels,selected_gesture)
emg_selected=emg[mask]
labels_selected=labels[mask]
repetition_selected=repetition[mask]
print(f"After gesture selection:")
print(f"EMG shape:    {emg_selected.shape}")
print(f"Labels shape: {labels_selected.shape}")
print(f"Gestures kept: {np.unique(labels_selected)}\n")
# check how many samples exist for each gesture
print("Sample count per gesture in the full dataset:")
for g in np.unique(labels):
    count = np.sum(labels == g)
    print(f"  Gesture {g:2d}: {count:6d} samples  ({count/fs:.1f} seconds)")

#apply bandpass filter
def bandpass_filter(signal, lowcut=20, highcut=500, fs=200, order=4):
    nyquist = fs / 2
    low = lowcut / nyquist
    high = min(highcut, nyquist * 0.99) / nyquist
    print(f"  Low cutoff:  {low:.4f}")
    print(f"  High cutoff: {high:.4f}")
    sos = butter(order, [low, high], btype='bandpass', output='sos')
    filtered= sosfiltfilt (sos , signal , axis=0)
    return filtered 


print(f"\nBandpass filter complete")
print(f"Input shape:  {emg_selected.shape}")
emg_bandpass = bandpass_filter(emg_selected,
                                lowcut=20,
                                highcut=500,
                                fs=fs,
                                order=4)
print(f"Output shape: {emg_bandpass.shape}")
print(f"Shape unchanged: {emg_selected.shape == emg_bandpass.shape}")

#apply notch filter 

from scipy.signal import iirnotch
def notch_filter(signal,notch_freq=50,quality_factor=30,fs=200):
    b,a = iirnotch(notch_freq/(fs/2),30  )
    filtered=filtfilt(b,a,signal,axis=0)
    return filtered
print("Applying notch filter (removing 50Hz powerline interference)...")
emg_notched = notch_filter(emg_bandpass, 
                            notch_freq=50, 
                            quality_factor=30, 
                            fs=fs)

print(f"Notch filter complete")
print(f"Input shape:  {emg_bandpass.shape}")
print(f"Output shape: {emg_notched.shape}")
print(f"Shape unchanged: {emg_bandpass.shape == emg_notched.shape}")

#raw vs filtered 
plot_samples = 2 * fs        
channel      = 1    
time = np.arange(plot_samples) / fs
fig, axes = plt.subplots(3, 1, figsize=(12, 8))
axes[0].plot(time, emg_selected[:plot_samples, channel],
             color='steelblue', linewidth=0.8)
axes[0].set_title('Raw EMG (before filtering)')
axes[0].set_ylabel('Amplitude (mV)')
axes[1].plot(time, emg_bandpass[:plot_samples, channel],
             color='darkorange', linewidth=0.8)
axes[1].set_title('After bandpass filter (20–99 Hz) — drift and high freq noise removed')
axes[1].set_ylabel('Amplitude (mV)')
axes[2].plot(time, emg_notched[:plot_samples, channel],
             color='green', linewidth=0.8)
axes[2].set_title('After notch filter (50 Hz removed) — final clean signal')
axes[2].set_ylabel('Amplitude (mV)')
axes[2].set_xlabel('Time (seconds)')

plt.suptitle('Effect of filtering on EMG signal — Channel 2', 
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

#normalization 
emg_mean = np.mean(emg_notched, axis=0)
emg_std  = np.std(emg_notched, axis=0)
emg_normalized = (emg_notched - emg_mean) / emg_std
print(f"Normalization complete")    
print(f"\nAfter normalization:")
print(f"Mean per channel (should be ~0): {np.round(emg_normalized.mean(axis=0), 4)}")
print(f"Std per channel  (should be ~1): {np.round(emg_normalized.std(axis=0), 4)}")
print(f"Shape unchanged: {emg_normalized.shape == emg_notched.shape}")

#windowing 
window_size = int(0.250 * fs)
step_size   = int(0.125 * fs)

print(f"Window size: {window_size} samples ({window_size/fs*1000:.0f}ms)")
print(f"Step size:   {step_size} samples ({step_size/fs*1000:.0f}ms)")

windows    = []
win_labels = []
win_reps   = []

n_windows = (len(emg_normalized) - window_size) // step_size

for i in range(n_windows):
    start = i * step_size
    end   = start + window_size
    
    window         = emg_normalized[start:end, :]
    window_labels  = labels_selected[start:end]
    
    unique, counts = np.unique(window_labels, return_counts=True)
    majority_label = unique[np.argmax(counts)]
    
    mid = start + window_size // 2
    rep = repetition_selected[mid]
    
    windows.append(window)
    win_labels.append(majority_label)
    win_reps.append(rep)

windows    = np.array(windows)
win_labels = np.array(win_labels)
win_reps   = np.array(win_reps)

print(f"\nWindowing complete:")
print(f"Windows shape:     {windows.shape}")
print(f"Labels shape:      {win_labels.shape}")
print(f"Repetitions shape: {win_reps.shape}")
print(f"\nSamples per gesture after windowing:")
for g in [1, 2, 6, 7]:
    count = np.sum(win_labels == g)
    print(f"  Gesture {g}: {count} windows")
