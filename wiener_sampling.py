import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, windows

# synthesize time series from PSD
def synthesize_from_psd(S, fs, N, rng=None):
    if rng is None: 
        rng = np.random.default_rng()
    f= np.fft.fftfreq(N, d=1/fs)
    w = 2* np.pi *f
    Svals = S(w)
    X = np.zeros(N, dtype=np.complex128)
    pos = np.arange(1, N//2)
    Z = (rng.standard_normal(pos.size) + 1j*rng.standard_normal(pos.size)) / np.sqrt(2.0)
    X[pos] = np.sqrt(fs * N * Svals[pos]) * Z
    X[-pos] = np.conj(X[pos])             
    return np.fft.ifft(X).real

# helper to generate all plots. Generate more than is needed in order to avoid circular artifacts
def generate_xy(Sxx, Snn, fs, N_keep, margin_sec=3.0, rng=None):
    M = int(round(margin_sec * fs))
    N = N_keep + 2*M
    x_full = synthesize_from_psd(Sxx, fs, N, rng)
    n_full = synthesize_from_psd(Snn, fs, N, rng)
    y_full = x_full + n_full
    sl = slice(M, M + N_keep)
    return x_full[sl], n_full[sl], y_full[sl]

# pick Welch nperseg such that eight Fourier bins span the target linewidth
def pick_nperseg(fs, gamma_rad_s, Nt, target_bins_across=8):
    d_omega = gamma_rad_s / target_bins_across
    df = d_omega / (2*np.pi)
    nperseg = int(np.ceil(fs / df))
    return min(nperseg, Nt)  # cannot exceed record length

# Welch method with 87.5%% overlap 
def welch_on_grid(x, y, fs, gamma_rad_s, w_grid, Nt):
    nperseg = pick_nperseg(fs, gamma_rad_s, Nt, target_bins_across=8)
    noverlap = int(0.875 * nperseg)
    nfft = w_grid.size  # <- force same grid length
    taper = windows.dpss(nperseg, NW=2.5, Kmax=1, sym=False)[0]

    f, Syy = welch(y, fs=fs, window=taper, nperseg=nperseg,
                   noverlap=noverlap, nfft=nfft, return_onesided=False,
                   scaling='density', detrend=False, average='mean')

    return Syy_to_func(w_grid, Syy)

# helper to use Syy as a function
def Syy_to_func(w_grid, values):
    idx = np.argsort(w_grid)
    ws = w_grid[idx]; vs = values[idx]
    def f(w): return np.interp(w, ws, vs)
    return f

# Constructing the causal wiener filter as outlined in the paper. Constructing the eigenbasis and calculating explicit coefficients. Solving Th=s, where T - toeplitz matrix, s - coefficients. 
def eigen_solve(K, N, Sxy_func, Syy_func, w, tan_freq=1):

    def Phi(k, w_):
        return (1/np.sqrt(np.pi)) / (1 - 1j*w_/tan_freq) * \
               ((1 + 1j*w_/tan_freq) / (1 - 1j*w_/tan_freq))**k

    du = 2*np.pi / N
    u  = -np.pi + (np.arange(N) + 0.5) * du
    w_tan = tan_freq * np.tan(u/2)

    Syy_u = Syy_func(w_tan)
    Sxy_u = Sxy_func(w_tan) * (np.sqrt(np.pi) * np.exp(1j*u/2) / np.cos(u/2))

    phase = ((-1)**np.arange(N)) * np.exp(1j * np.pi * np.arange(N) / N)

    t_all = np.fft.ifft(Syy_u) * phase
    s_all = np.fft.ifft(Sxy_u) * phase 

    tvals = t_all[:K+1] 
    s     = s_all[:K+1] 
    idx = np.abs(np.subtract.outer(np.arange(K+1), np.arange(K+1)))
    T = tvals[idx]

    h_coef, *_ = np.linalg.lstsq(T, s, rcond=None)
    k = np.arange(K+1)[:, None]
    Phi_mat = Phi(k, w)
    return h_coef @ Phi_mat

# calculating wiener filter on a lattice 
def grid_solve(w, Sxy_func, Syy_func):
    Sxy = np.array(Sxy_func(w), dtype=complex)
    Syy = np.array(Syy_func(w), dtype=complex)
    dw = np.abs(w[0] - w[1])
    I = np.eye(np.size(w))
    diff = w[:, None] - w[None, :] 
    H = dw/np.pi * np.where(diff == 0.0, 0.0, 1.0/diff)
    Sxy_real = np.real(Sxy)
    Sxy_imag = np.imag(Sxy)
    M = np.diag(Syy) - H @ (Syy[:, None] * H)
    rhs = Sxy_real - (H @ Sxy_imag)
    h_real = np.linalg.solve(M, rhs)     
    h_imag = H @ h_real              
    h = h_real + 1j*h_imag
    return h

