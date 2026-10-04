import glob
import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.signal import butter, iirnotch, filtfilt
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

WIN, HOP = 300, 150      # 샘플 단위 (1000Hz -> 300ms, 150ms)

def make_windows(x, win=WIN, hop=HOP):
    """x: (시간, 채널) -> (윈도우수, win, 채널)"""
    n = (len(x) - win) // hop + 1
    return np.stack([x[i*hop : i*hop+win] for i in range(n)])

w = make_windows(xf)

def minmax(w, eps=1e-8):
    """w: (윈도우수, 시간, 채널)"""
    mn = w.min(axis=(1, 2), keepdims=True)
    mx = w.max(axis=(1, 2), keepdims=True)
    return (w - mn) / (mx - mn + eps)

wn = minmax(w)

SCALES = np.arange(1, 33)        

def to_cwt(one_window, wavelet='morl'):
    """(300, 2) -> (3, 32, 300)"""
    maps = []
    for ch in range(one_window.shape[1]):
        coef, _ = pywt.cwt(one_window[:, ch], SCALES, wavelet)
        maps.append(np.abs(coef))        
    maps.append((maps[0] + maps[1]) / 2)   
    return np.stack(maps).astype(np.float32)

t = to_cwt(wn[0])
print('입력 텐서 모양:', t.shape)    
plt.imshow(t[0])
plt.show()