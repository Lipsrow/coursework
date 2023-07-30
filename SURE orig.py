import numpy as np
from numpy import linalg as lg
import matplotlib.pyplot as plt
from pywt import wavedec, waverec

def SURE_tresholding(signal, sigma, level):
    X = lg.norm(signal)
    coeffs = wavedec(signal, st, level=level)
    cA = coeffs[0]
    signal_det = coeffs[1:]
    det_coefs = np.array([])
    for i in signal_det:
        det_coefs = np.append(det_coefs, i)
    N = len(det_coefs)
    e = sigma ** 2 * np.sqrt(N) * np.log(N) ** (1.5)
    T_U = sigma * np.sqrt(2 * np.log(N))
    if X - N * sigma <= e:
        T = T_U
    else:
        signal_s = np.sort(det_coefs)
        S = signal_s[signal_s <= T_U]
        T = S[-1]

    signal_thresh = det_coefs.copy()
    signal_thresh[np.abs(det_coefs) < T] = 0
    cDs = [cA]
    lens = list(map(len, signal_det))
    el = 0
    for i in lens:
        a = np.array([])
        for j in range(i):
            a = np.append(a, signal_thresh[el])
            el += 1
        cDs.append(a)
    signal_thresh = waverec(cDs, st)
    return signal_thresh


def plot(x, signal, signal_tresh):
    fig, axes = plt.subplots(2, 1)
    axes[0].plot(x, signal)
    axes[1].plot(x, signal_tresh)
    axes[0].grid()
    axes[1].grid()
    plt.suptitle("Метод SURE для f(x) = sin(x) + sin(2x) + sin(3x), σ = 0.3, L = 3")
    axes[0].set(title="Сигнал до обработки")
    axes[1].set(title="Сигнал после обработки")
    plt.subplots_adjust(hspace=0.4)
    plt.show()


np.random.seed(1)
st = 'db6'
sigma = 0.3  # cреднеквадратичное отклонение шума
N = 2 ** 10  # количество точек
x = np.linspace(-5, 5, N)
F =  np.sin(x) + np.sin(2 * x) + np.sin(3 * x)
W = np.random.normal(0, sigma, N)  # значения шума
signal = F + W  # значения сигнала с шумом
signal_tresh = SURE_tresholding(signal, sigma, 3)  # обработанный сигнал
plot(x, signal, signal_tresh)
