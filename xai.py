"""
Phase 5 — Explainability (XAI)
SHAP DeepExplainer and LIME tabular explanations.
Generates global importance, local prediction explanations.
"""

import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import shap
import lime
import lime.lime_tabular
from preprocessing import windows, win_labels, win_reps, fs
from feature_extraction import extract_features
from sklearn.preprocessing import LabelEncoder



# ── RECREATE MODEL ────────────────────────────────────────────────────
class EMGClassifier(nn.Module):
    def __init__(self, input_size=80, num_classes=4):
        super(EMGClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 4)
        )
    def forward(self, x):
        return self.network(x)

# ── LOAD MODEL ────────────────────────────────────────────────────────
model = EMGClassifier(input_size=80, num_classes=4)

model.eval()
print("Model loaded\n")

# ── PREPARE DATA ──────────────────────────────────────────────────────
X    = extract_features(windows)
y    = win_labels
reps = win_reps

le         = LabelEncoder()
y_enc      = le.fit_transform(y)
train_mask = reps <= 4
test_mask  = reps > 4
X_train    = X[train_mask]
X_test     = X[test_mask]
y_test_enc = y_enc[test_mask]

# ── FEATURE NAMES ─────────────────────────────────────────────────────
feature_names = []
for feat in ['RMS', 'MAV', 'ZCR', 'WL', 'VAR']:
    for ch in range(1, 17):
        feature_names.append(f'{feat}_ch{ch}')

gesture_names = {0: 'index flexion',
                 1: 'ring flexion',
                 2: 'index extension',
                 3: 'wrist extension'}

print(f"Test samples:  {X_test.shape[0]}")
print(f"Feature names: {len(feature_names)}\n")

# ── TENSORS ───────────────────────────────────────────────────────────
X_train_t = torch.FloatTensor(X_train)
X_test_t  = torch.FloatTensor(X_test)

# ── GET MODEL PREDICTIONS ─────────────────────────────────────────────
with torch.no_grad():
    outputs     = model(X_test_t)
    model_preds = outputs.argmax(dim=1).numpy()

print("Prediction distribution on test set:")
for class_idx in range(4):
    predicted_count = np.sum(model_preds == class_idx)
    actual_count    = np.sum(y_test_enc == class_idx)
    print(f"  {gesture_names[class_idx]:20s} — "
          f"actual: {actual_count:3d} | predicted: {predicted_count:3d}")
print()

# ── SHAP ──────────────────────────────────────────────────────────────
print("Setting up SHAP DeepExplainer...")
background  = X_train_t[:100]
explainer   = shap.DeepExplainer(model, background)

print("Computing SHAP values — this takes a minute...")
shap_values = explainer.shap_values(X_test_t)

# reshape to (4, n_samples, 80)
shap_values = np.array(shap_values)
if shap_values.ndim == 3 and shap_values.shape[2] == 4:
    shap_values = shap_values.transpose(2, 0, 1)
elif shap_values.ndim == 3 and shap_values.shape[0] == 4:
    pass

print(f"SHAP values shape: {shap_values.shape}\n")

# ── PLOT 1: GLOBAL FEATURE IMPORTANCE ─────────────────────────────────
print("Saving Plot 1 — global SHAP importance...")

for class_idx in range(4):
    fig, ax = plt.subplots(figsize=(10, 7))

    mean_shap = np.abs(shap_values[class_idx]).mean(axis=0)
    top_idx   = np.argsort(mean_shap)[-15:]
    top_vals  = mean_shap[top_idx]
    top_names = [feature_names[i] for i in top_idx]

    ax.barh(range(15), top_vals, color='steelblue', alpha=0.8)
    ax.set_yticks(range(15))
    ax.set_yticklabels(top_names, fontsize=10)
    ax.set_xlabel('Mean absolute SHAP value', fontsize=11)
    ax.set_title(f'Top 15 features — {gesture_names[class_idx]}',
                 fontsize=13, pad=12)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    plt.show()


# ── PLOT 2: SINGLE PREDICTION EXPLANATION ALL 4 GESTURES ──────────────
print("Saving Plot 2 — single prediction SHAP...")

for class_idx in range(4):
    fig, ax = plt.subplots(figsize=(10, 8))

    # try correct predictions first
    correct = np.where((y_test_enc == class_idx) &
                       (model_preds == class_idx))[0]

    if len(correct) == 0:
        correct      = np.where(y_test_enc == class_idx)[0]
        title_suffix = '(model was wrong — showing explanation anyway)'
    else:
        title_suffix = '(correct prediction)'

    if len(correct) == 0:
        ax.set_title(f'No samples found — {gesture_names[class_idx]}')
        plt.close()
        continue

    sample_idx   = correct[0]
    sample_shap  = shap_values[class_idx][sample_idx]
    sorted_idx   = np.argsort(np.abs(sample_shap))[-20:]
    sorted_vals  = sample_shap[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]
    colors       = ['coral' if v < 0 else 'steelblue' for v in sorted_vals]

    ax.barh(range(20), sorted_vals, color=colors, alpha=0.8)
    ax.set_yticks(range(20))
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('SHAP value (blue = toward | red = away)', fontsize=11)
    ax.set_title(f'Single prediction — {gesture_names[class_idx]}\n'
                 f'{title_suffix}', fontsize=12, pad=12)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.show()


# ── PLOT 3: CONFUSION EXPLANATION ─────────────────────────────────────
print("Saving Plot 3 — confusion explanation...")

correct_idx  = np.where((y_test_enc == 0) & (model_preds == 0))[0]
confused_idx = np.where((y_test_enc == 0) & (model_preds == 1))[0]

print(f"  Correct index flexion:    {len(correct_idx)}")
print(f"  Confused as ring flexion: {len(confused_idx)}")

if len(correct_idx) > 0 and len(confused_idx) > 0:
    correct_shap  = np.abs(shap_values[0][correct_idx]).mean(axis=0)
    confused_shap = np.abs(shap_values[1][confused_idx]).mean(axis=0)
    diff          = np.abs(correct_shap - confused_shap)
    top_diff_idx  = np.argsort(diff)[-10:]
    x             = np.arange(10)
    width         = 0.35

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.bar(x - width/2, correct_shap[top_diff_idx],
           width, label='correct index flexion',
           color='steelblue', alpha=0.8)
    ax.bar(x + width/2, confused_shap[top_diff_idx],
           width, label='confused as ring flexion',
           color='coral', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([feature_names[i] for i in top_diff_idx],
                        rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Mean absolute SHAP value', fontsize=11)
    ax.set_title('Why does index flexion get confused with ring flexion?\n'
                 'Features that differ between correct and confused predictions',
                 fontsize=12, pad=12)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()
    print("  plot was plotted successfully")
else:
    print("  Skipping — not enough samples for comparison")

print()

# ── LIME ──────────────────────────────────────────────────────────────
print("Setting up LIME explainer...")

def predict_fn(x):
    tensor = torch.FloatTensor(x)
    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.nn.functional.softmax(outputs, dim=1)
    return probs.numpy()

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data = X_train,
    feature_names = feature_names,
    class_names   = list(gesture_names.values()),
    mode          = 'classification'
)
print("LIME explainer created\n")

# ── PLOT 4: LIME EXPLANATION ONE PLOT PER GESTURE ─────────────────────
print("Saving Plot 4 — LIME explanations...")

for class_idx in range(4):
    correct = np.where((y_test_enc == class_idx) &
                       (model_preds == class_idx))[0]

    if len(correct) == 0:
        correct      = np.where(y_test_enc == class_idx)[0]
        title_suffix = '(model was wrong)'
    else:
        title_suffix = '(correct prediction)'

    if len(correct) == 0:
        print(f"  No samples for {gesture_names[class_idx]} — skipping")
        continue

    sample_idx = correct[0]

    exp = lime_explainer.explain_instance(
        data_row     = X_test[sample_idx],
        predict_fn   = predict_fn,
        num_features = 15,
        num_samples  = 500,
        top_labels   = 4
    )

    if class_idx not in exp.available_labels():
        print(f"  LIME label {class_idx} not available — skipping")
        continue

    lime_vals = exp.as_list(label=class_idx)
    names     = [x[0] for x in lime_vals]
    vals      = [x[1] for x in lime_vals]
    colors    = ['steelblue' if v > 0 else 'coral' for v in vals]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(vals)), vals, color=colors, alpha=0.8)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names, fontsize=9)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('LIME weight (blue = supports | red = contradicts)',
                  fontsize=11)
    ax.set_title(f'LIME explanation — {gesture_names[class_idx]}\n'
                 f'{title_suffix}', fontsize=12, pad=12)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.show()

print("Phase 5 complete.")
