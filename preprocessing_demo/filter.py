import glob
import numpy as np
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
print('전:', x.std(), ' 후:', xf.std())