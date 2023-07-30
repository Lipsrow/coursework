from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from pywt import wavedec, waverec



def FDR_tresholding(signal, sigma, q, level):
    coeffs = wavedec(signal, st, level=level)
    cA = coeffs[0]
    signal_det = coeffs[1:]
    det_coefs = np.array([])
    for i in signal_det:
        det_coefs = np.append(det_coefs, i)
    N = len(det_coefs)

    def p_value(x):
        return 2 * (1 - norm.cdf(x / sigma))

    P = np.vectorize(p_value)(det_coefs)
    k = 0
    for i in range(N):
        if P[i] <= i * q / N:
            k = i
    T = sigma * norm.ppf(1 - P[k] / 2)
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


def plot(x, signal, signal_tresh, L):
    fig, axes = plt.subplots(2, 1)
    axes[0].plot(x, signal)
    axes[1].plot(x, signal_tresh)
    axes[0].grid()
    axes[1].grid()
    plt.suptitle(f"Метод FDR для f(x) = sin(x) + sin(2x) + sin(3x), σ = 0.3, L = {L}")
    axes[0].set(title="Сигнал до обработки")
    axes[1].set(title="Сигнал после обработки")
    plt.subplots_adjust(hspace=0.4)
    plt.show()

np.random.seed(0)
st = 'db38'
L = 3
q = 0.005  # ошибка первого рода
sigma = 0.3  # среднеквадратичное отклонение шума
N = 2 ** 10  # количество точек
x = np.linspace(-5, 5, N)
F =  np.sin(x) + np.sin(2 * x) + np.sin(3 * x) # значения сигнала без шума
W = np.random.normal(0, sigma, N)  # значения шума
signal = F + W  # значения сигнала с шумом
signal_tresh = FDR_tresholding(signal, sigma, q, L)  # обработанный сигнал
plot(x, signal, signal_tresh, L)
