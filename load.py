import scipy.io
import numpy as np
import matplotlib.pyplot as plt

data = scipy.io.loadmat(r'D:\Users\Acer\Desktop\hand gesture classification\S1_E1_A1.mat')

print("Keys in the file:")
for key in data.keys():
    print(" ", key)

emg=data['emg']
print('shape of emg:', emg.shape)
labels = data['restimulus'].squeeze()
print('shape of labels:', labels.shape)
repetition = data['repetition'].squeeze()
print('shape of repetition:', repetition.shape)
fs = int(data['frequency'].squeeze())

#print(f"Unique gestures: {np.unique(labels)}")
print(f"Unique repetitions: {np.unique(repetition)}")
print(f"Recording duration: {emg.shape[0] / fs:.1f} seconds")

# ── VISUALIZE RAW EMG ──
n_samples = 5 * fs


time = np.arange(n_samples) / fs 
fig, axes = plt.subplots(4, 1, figsize=(12, 8))

for i in range(4):
   
    axes[i].plot(time, emg[:n_samples, i], 
                 color='steelblue', 
                 linewidth=0.8)
    
    axes[i].set_ylabel(f'Ch {i+1} (mV)')
    
    
    ax2 = axes[i].twinx()

    ax2.plot(time, labels[:n_samples], 
             color='red', 
             linewidth=0.6, 
             alpha=0.5)
    ax2.set_ylabel('gesture', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

axes[0].set_title('Raw EMG signal — first 5 seconds (blue) with gesture label (red)')
axes[-1].set_xlabel('Time (seconds)')

plt.tight_layout()

plt.show()

# ── FIND WHERE GESTURES ACTUALLY HAPPEN ───────────────────────────────


gesture1_start = np.where(labels == 1)[0][0]

#print(f"Gesture 1 starts at sample: {gesture1_start}")
#print(f"Gesture 1 starts at time: {gesture1_start / fs:.2f} seconds")

# 1 second before it starts to 2 seconds after
start = gesture1_start - 1 * fs   # 1 second before
end   = gesture1_start + 2 * fs   # 2 seconds after

time2 = np.arange(end - start) / fs

fig, axes = plt.subplots(4, 1, figsize=(12, 8))

for i in range(4):
    axes[i].plot(time2, emg[start:end, i],
                 color='steelblue', linewidth=0.8)
    axes[i].set_ylabel(f'Ch {i+1} (mV)')

    ax2 = axes[i].twinx()
    ax2.plot(time2, labels[start:end],
             color='red', linewidth=1.2, alpha=0.7)
    ax2.set_ylabel('gesture', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

axes[0].set_title('Raw EMG — moment gesture 1 begins (red line jumps from 0 to 1)')
axes[-1].set_xlabel('Time (seconds)')
plt.tight_layout()
plt.show()