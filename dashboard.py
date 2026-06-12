"""
Phase 6 — Live XAI Dashboard
Real-time animated dashboard showing EMG signal, gesture prediction,
confidence scores, and SHAP explanation updating every 1 second.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import torch
import torch.nn as nn
import shap
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
test_mask  = reps > 4
X_test     = X[test_mask]
y_test_enc = y_enc[test_mask]
windows_test = windows[test_mask]

train_mask = reps <= 4
X_train    = X[train_mask]

# ── FEATURE NAMES ─────────────────────────────────────────────────────
feature_names = []
for feat in ['RMS', 'MAV', 'ZCR', 'WL', 'VAR']:
    for ch in range(1, 17):
        feature_names.append(f'{feat}_ch{ch}')

gesture_names  = ['index flexion', 'ring flexion',
                  'index extension', 'wrist extension']
gesture_colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']

print("Precomputing SHAP values for test set...")
X_train_t   = torch.FloatTensor(X_train)
X_test_t    = torch.FloatTensor(X_test)
background  = X_train_t[:100]
explainer   = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(X_test_t)
shap_values = np.array(shap_values)
if shap_values.ndim == 3 and shap_values.shape[2] == 4:
    shap_values = shap_values.transpose(2, 0, 1)
print(f"SHAP precomputed. Shape: {shap_values.shape}\n")

# ── GET ALL PREDICTIONS ───────────────────────────────────────────────
with torch.no_grad():
    outputs      = model(X_test_t)
    probs        = torch.nn.functional.softmax(outputs, dim=1).numpy()
    model_preds  = outputs.argmax(dim=1).numpy()

print(f"Total test windows available: {len(X_test)}")
print("Starting dashboard...\n")

# ── DASHBOARD SETUP ───────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('#1a1a2e')
fig.suptitle('EMG Gesture Classification — Live XAI Dashboard',
             fontsize=16, color='white', fontweight='bold', y=0.99)

# grid layout: 2 rows, 3 columns
gs = gridspec.GridSpec(2, 3,
                       figure=fig,
                       hspace=0.45,
                       wspace=0.35,
                       left=0.06, right=0.97,
                       top=0.88, bottom=0.08)

ax_signal  = fig.add_subplot(gs[0, :2])   # top left — EMG signal
ax_shap    = fig.add_subplot(gs[:, 2])    # right — SHAP bars
ax_pred    = fig.add_subplot(gs[1, 0])    # bottom left — prediction
ax_conf    = fig.add_subplot(gs[1, 1])    # bottom middle — confidence

# ── STYLE ALL AXES ────────────────────────────────────────────────────
bg_color   = '#16213e'
text_color = 'white'

for ax in [ax_signal, ax_shap, ax_pred, ax_conf]:
    ax.set_facecolor(bg_color)
    ax.tick_params(colors=text_color, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444466')

# ── SIGNAL PANEL ──────────────────────────────────────────────────────
ax_signal.set_title('Live EMG Signal — Channel 1',
                     color=text_color, fontsize=11, pad=8)
ax_signal.set_xlabel('Sample', color=text_color, fontsize=9)
ax_signal.set_ylabel('Amplitude', color=text_color, fontsize=9)
line_signal, = ax_signal.plot([], [], color='#00d4ff',
                               linewidth=0.8, alpha=0.9)
ax_signal.set_xlim(0, 50)
ax_signal.set_ylim(-3, 3)

# true label text on signal plot
true_label_text = ax_signal.text(
    0.02, 0.92, '', transform=ax_signal.transAxes,
    color='#aaaacc', fontsize=9)

# ── SHAP PANEL ────────────────────────────────────────────────────────
ax_shap.set_title('SHAP Explanation\n(top 10 features)',
                   color=text_color, fontsize=11, pad=8)
ax_shap.set_xlabel('SHAP value', color=text_color, fontsize=9)

# initialize empty shap bars
shap_bars     = ax_shap.barh(range(10), [0]*10,
                              color='steelblue', alpha=0.8)
shap_yticks   = ax_shap.set_yticks(range(10))
shap_ylabels  = ax_shap.set_yticklabels([''] * 10,
                                          fontsize=7,
                                          color=text_color)
ax_shap.axvline(x=0, color='white', linewidth=0.5, alpha=0.5)
ax_shap.set_xlim(-0.3, 0.3)

# ── PREDICTION PANEL ──────────────────────────────────────────────────
ax_pred.set_title('Predicted Gesture',
                   color=text_color, fontsize=11, pad=8)
ax_pred.axis('off')

pred_text = ax_pred.text(
    0.5, 0.55, '...', transform=ax_pred.transAxes,
    ha='center', va='center',
    fontsize=16, fontweight='bold', color='white')

conf_text = ax_pred.text(
    0.5, 0.25, '', transform=ax_pred.transAxes,
    ha='center', va='center',
    fontsize=12, color='#aaaacc')

correct_text = ax_pred.text(
    0.5, 0.08, '', transform=ax_pred.transAxes,
    ha='center', va='center',
    fontsize=10, color='white')

# ── CONFIDENCE PANEL ──────────────────────────────────────────────────
ax_conf.set_title('Confidence Scores',
                   color=text_color, fontsize=11, pad=8)
ax_conf.set_xlim(0, 1)
ax_conf.set_ylim(-0.5, 3.5)
ax_conf.set_yticks(range(4))
ax_conf.set_yticklabels(gesture_names, color=text_color, fontsize=8)
ax_conf.set_xlabel('Probability', color=text_color, fontsize=9)
ax_conf.axvline(x=0.25, color='#444466',
                 linewidth=0.8, linestyle='--', alpha=0.7)

conf_bars = ax_conf.barh(range(4), [0.25]*4,
                          color=gesture_colors, alpha=0.8)

conf_value_texts = []
for i in range(4):
    t = ax_conf.text(0.27, i, '25%',
                      va='center', fontsize=8, color=text_color)
    conf_value_texts.append(t)

# ── WINDOW COUNTER ────────────────────────────────────────────────────
counter_text = fig.text(
    0.5, 0.005,
    'Window 0 / 0',
    ha='center', fontsize=9,
    color='#aaaacc')

# ── ANIMATION FUNCTION ────────────────────────────────────────────────
current_idx = [0]

def update(frame):
    idx = current_idx[0] % len(X_test)
    current_idx[0] += 1

    # ── update signal panel ───────────────────────────────────────────
    signal_data = windows_test[idx, :, 0]
    line_signal.set_data(range(50), signal_data)
    true_gesture = gesture_names[y_test_enc[idx]]
    true_label_text.set_text(f'True: {true_gesture}')

    # ── update prediction panel ───────────────────────────────────────
    pred_class   = model_preds[idx]
    pred_name    = gesture_names[pred_class]
    pred_conf    = probs[idx][pred_class] * 100
    pred_color   = gesture_colors[pred_class]

    pred_text.set_text(pred_name.upper())
    pred_text.set_color(pred_color)
    conf_text.set_text(f'{pred_conf:.1f}% confident')

    is_correct = pred_class == y_test_enc[idx]
    correct_text.set_text('✓ correct' if is_correct else '✗ wrong')
    correct_text.set_color('#2ECC71' if is_correct else '#E74C3C')

    # ── update confidence bars ────────────────────────────────────────
    for i, (bar, txt) in enumerate(zip(conf_bars, conf_value_texts)):
        prob = probs[idx][i]
        bar.set_width(prob)
        bar.set_alpha(1.0 if i == pred_class else 0.4)
        txt.set_text(f'{prob*100:.1f}%')
        txt.set_x(prob + 0.02)

    # ── update shap panel ─────────────────────────────────────────────
    sample_shap  = shap_values[pred_class][idx]
    sorted_idx   = np.argsort(np.abs(sample_shap))[-10:]
    sorted_vals  = sample_shap[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]
    colors       = [gesture_colors[pred_class] if v > 0
                    else '#E74C3C' for v in sorted_vals]

    for i, (bar, val, name, col) in enumerate(
            zip(shap_bars, sorted_vals, sorted_names, colors)):
        bar.set_width(val)
        bar.set_color(col)
        bar.set_alpha(0.8)

    ax_shap.set_yticklabels(sorted_names,
                              fontsize=7,
                              color=text_color)
    ax_shap.set_xlim(
        min(-0.3, sorted_vals.min() * 1.2),
        max(0.3,  sorted_vals.max() * 1.2))

    # ── update counter ────────────────────────────────────────────────
    counter_text.set_text(
        f'Window {idx+1} / {len(X_test)} | '
        f'True: {true_gesture} | '
        f'Predicted: {pred_name}')

    return ([line_signal, pred_text, conf_text, correct_text,
             true_label_text, counter_text] +
            list(shap_bars) + list(conf_bars) +
            list(conf_value_texts))

# ── RUN ANIMATION ─────────────────────────────────────────────────────
ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(X_test),
    interval=1000,        # 1000ms between frames
    blit=False,
    repeat=True
)

plt.show()