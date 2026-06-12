"""
Phase 4 — PyTorch Classifier
MLP architecture: 80 → 128 → 64 → 32 → 4
Train on repetitions 1-4, test on repetitions 5-6.
Final test accuracy: 68%
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from logger import start_logger
from preprocessing import windows, win_labels, win_reps, fs
from feature_extraction import extract_features


# ── FEATURE EXTRACTION ────────────────────────────────────────────────
X    = extract_features(windows)
y    = win_labels
reps = win_reps

# ── STEP 1: TRAIN/TEST SPLIT ──────────────────────────────────────────
train_mask = (reps <= 4)
test_mask  = (reps > 4)

X_train = X[train_mask]
X_test  = X[test_mask]
y_train = y[train_mask]
y_test  = y[test_mask]

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")
print(f"Feature size:     {X_train.shape[1]}")

# ── STEP 2: ENCODE LABELS ─────────────────────────────────────────────

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc  = le.transform(y_test)

print(f"\nOriginal labels:  {le.classes_}")
print(f"Encoded labels:   {np.unique(y_train_enc)}")

# ── STEP 3: CONVERT TO PYTORCH TENSORS ───────────────────────────────
X_train_t = torch.FloatTensor(X_train)
X_test_t  = torch.FloatTensor(X_test)
y_train_t = torch.LongTensor(y_train_enc)
y_test_t  = torch.LongTensor(y_test_enc)

# ── STEP 4: CREATE DATALOADERS ────────────────────────────────────────
train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset  = TensorDataset(X_test_t,  y_test_t)

train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True , drop_last=True)
test_loader   = DataLoader(test_dataset,  batch_size=32, shuffle=False)

print(f"\nTraining batches: {len(train_loader)}")
print(f"Testing batches:  {len(test_loader)}")

# ── STEP 5: BUILD NEURAL NETWORK ──────────────────────────────────────
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
            
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

model = EMGClassifier(input_size=80, num_classes=4)
print(f"\nModel architecture:")
print(model)

# ── STEP 6: TRAINING SETUP ────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

n_epochs  = 100

train_losses = []
test_losses  = []
train_accs   = []
test_accs    = []

# ── STEP 7: TRAINING LOOP ─────────────────────────────────────────────
print(f"\nTraining for {n_epochs} epochs...")
print("-" * 50)

for epoch in range(n_epochs):
    
    # ── training phase ────────────────────────────────────────────────
    model.train()
    batch_losses = []
    correct      = 0
    total        = 0
    
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs  = model(X_batch)
        loss     = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        batch_losses.append(loss.item())
        predicted = outputs.argmax(dim=1)
        correct  += (predicted == y_batch).sum().item()
        total    += y_batch.size(0)
    
    train_loss = np.mean(batch_losses)
    train_acc  = correct / total

    # ── evaluation phase ──────────────────────────────────────────────
    model.eval()
    batch_losses = []
    correct      = 0
    total        = 0
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs   = model(X_batch)
            loss      = criterion(outputs, y_batch)
            batch_losses.append(loss.item())
            predicted = outputs.argmax(dim=1)
            correct  += (predicted == y_batch).sum().item()
            total    += y_batch.size(0)
    
    test_loss = np.mean(batch_losses)
    test_acc  = correct / total
    
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    
    scheduler.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}/{n_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

# ── STEP 8: PLOT TRAINING CURVES ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(train_losses, label='train loss', color='steelblue')
axes[0].plot(test_losses,  label='test loss',  color='coral')
axes[0].set_title('Loss over epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()

axes[1].plot(train_accs, label='train accuracy', color='steelblue')
axes[1].plot(test_accs,  label='test accuracy',  color='coral')
axes[1].set_title('Accuracy over epochs')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()

plt.tight_layout()
plt.show()

# ── STEP 9: FINAL EVALUATION ──────────────────────────────────────────
model.eval()
all_preds  = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs   = model(X_batch)
        predicted = outputs.argmax(dim=1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(y_batch.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

gesture_names = ['index flexion', 'ring flexion',
                 'index extension', 'wrist extension']

print("\nClassification Report:")
print(classification_report(all_labels, all_preds,
                             target_names=gesture_names))

# ── STEP 10: CONFUSION MATRIX ─────────────────────────────────────────
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=gesture_names,
            yticklabels=gesture_names,
            cmap='Blues')
plt.title('Confusion matrix — gesture classification')
plt.ylabel('True gesture')
plt.xlabel('Predicted gesture')
plt.tight_layout()
plt.show()

