import glob
import numpy as np
import matplotlib.pyplot as plt

files = sorted(glob.glob('data/**/*.csv', recursive=True))
x = np.loadtxt(files[180], delimiter=',', skiprows=1)

# (시간, 채널) 형태로 통일
if x.shape[0] < x.shape[1]:
    x = x.T

# 샘플링 주파수 1000 Hz 기준 시간(초)
t = np.arange(len(x)) / 1000.0

fig, ax = plt.subplots(2, 1, figsize=(10, 4), sharex=True)

for ch in range(2):
    ax[ch].plot(t, x[:, ch], lw=0.6)
    ax[ch].set_ylabel(f'ch{ch + 1}')

ax[1].set_xlabel('time (s)')
plt.tight_layout()
plt.savefig('signal_example.png', dpi=120)
plt.show()