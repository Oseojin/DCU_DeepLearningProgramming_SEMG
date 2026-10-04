import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, iirnotch, filtfilt, welch
FS = 1000

def preprocess(x, fs=FS):
    # 1) 60 Hz 전원선 잡음 제거
    bn, an = iirnotch(60, 30, fs)
    x = filtfilt(bn, an, x, axis=0)
    # 2) 20~499 Hz 대역만 통과 (4차)
    b, a = butter(4, [20/(fs/2), 499/(fs/2)], btype='band')
    return filtfilt(b, a, x, axis=0)

files = sorted(glob.glob('data/**/*.csv', recursive=True))
x = np.loadtxt(files[180], delimiter=',', skiprows=1)
xf = preprocess(x)


f0, P0 = welch(x[:, 0],  fs=FS, nperseg=512)
f1, P1 = welch(xf[:, 0], fs=FS, nperseg=512)

plt.figure(figsize=(9, 3))
plt.semilogy(f0, P0, label='before', lw=0.8)
plt.semilogy(f1, P1, label='after',  lw=0.8)
plt.axvline(60, color='r', ls='--')
plt.xlim(0, 300); plt.legend()
plt.xlabel('Hz'); plt.ylabel('power')
plt.tight_layout(); plt.savefig('filter_check.png', dpi=120)
plt.show()