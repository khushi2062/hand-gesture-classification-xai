"""
Phase 3 — Feature Extraction
Extracts RMS, MAV, ZCR, WL, VAR per channel from each window.
Output: feature matrix X (710, 80), labels y (710,)
""" 
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import windows, win_labels, win_reps, fs
print(f"Windows loaded: {windows.shape}")
print(f"Labels loaded:  {win_labels.shape}")
print(f"Reps loaded:    {win_reps.shape}\n")

def rms(window):
    return np.sqrt(np.mean(window**2, axis=0))

def mav(window):
    return np.mean(np.abs(window), axis=0)

def zcr(window):
    signs     = np.sign(window)
    diff      = np.diff(signs, axis=0)
    crossings = np.sum(diff != 0, axis=0)
    return crossings

def wl(window):
    return np.sum(np.abs(np.diff(window, axis=0)), axis=0)

def var(window):
    return np.var(window, axis=0, ddof=1)

def extract_features(windows):
    n_windows = windows.shape[0]
    features  = np.zeros((n_windows, 80))
    for i in range(n_windows):
        window     = windows[i]
        features[i] = np.concatenate([rms(window),
                                       mav(window),
                                       zcr(window),
                                       wl(window),
                                       var(window)])
    return features

X    = extract_features(windows)
y    = win_labels
reps = win_reps

print(f"Feature matrix shape: {X.shape}")
print(f"Labels shape:         {y.shape}")
print(f"Correct: {X.shape == (len(windows), 80)}")


# ── VISUALIZE FEATURES PER GESTURE ────────────────────────────────────
gesture_names = {1: 'index flexion', 2: 'ring flexion',
                 6: 'index extension', 7: 'wrist extension'}

fig, axes = plt.subplots(1, 5, figsize=(15, 4))
feature_names = ['RMS', 'MAV', 'ZCR', 'WL', 'VAR']

for feat_idx, (ax, fname) in enumerate(zip(axes, feature_names)):
    for g in [1, 2, 6, 7]:
        mask   = y == g
        # mean of first channel for this feature across all windows of gesture g
        values = X[mask, feat_idx]
        ax.hist(values, alpha=0.5, label=gesture_names[g], bins=20)
    ax.set_title(fname)
    ax.set_xlabel('value')
    ax.legend(fontsize=6)

plt.suptitle('Feature distributions per gesture', fontsize=13)
plt.tight_layout()

plt.show()