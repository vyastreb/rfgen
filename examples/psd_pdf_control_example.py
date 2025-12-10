import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from numpy.fft import fftn, ifftn, fftfreq

from rfgen import arbitrary_pdf_psd_field, psd_radial_average

def main():
    N = 512

    # Target PDF
    c = 1.5
    loc = 0.0
    scale = 1.0
    def pdf_func(z):
        return stats.weibull_min.pdf(z, c=c, loc=loc, scale=scale)
    
    # Target PSD
    alpha = 0.001
    H = 0.5
    def psd_func(k):
        val = (alpha + k**2)**(-(1+H)) 
        return val

    # Generate field with prescribed PSD and PDF
    field, psd_scale = arbitrary_pdf_psd_field(dim=2, N=N, psd_func=psd_func, pdf_func=pdf_func, verbose=True, return_psd_scale=True)
    
    # Plot results
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Surface
    vmin = -0.
    vmax =  2.5
    im = ax[0].imshow(field - np.min(field), cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
    ax[0].set_title("Generated Surface (shifter)")

    # 2. PDF
    flat = field.flatten()
    flat -= np.min(flat)
    x_vals = np.linspace(min(flat), max(flat), 200)
    ax[1].hist(flat, bins=100, density=True, alpha=0.6, label="Result shifted")
    ax[1].plot(x_vals, pdf_func(x_vals), "r--", lw=2, label="Target")
    ax[1].set_title("PDF")
    ax[1].set_xlabel("Height")
    ax[1].set_ylabel("Probability Density")
    ax[1].legend()

    # 3. PSD
    k, psd = psd_radial_average(field)
    ax[2].loglog(k, psd, "v-", label="Result",  alpha=0.7)

    # Target PSD (scaled for shape comparison)
    k_target = np.linspace(k[1], k[-1], 500)
    target_vals = psd_func(k_target)
    # Scale factor to align peaks for visual comparison
    scale = 1.0
    if np.max(target_vals) > 0:
        scale = np.max(psd) / np.max(target_vals)
    
    ax[2].loglog(k_target, target_vals * scale, "-", color="r", lw=3, label="Target (scaled)", alpha=0.5)
    ax[2].set_title("PSD (radial)")
    ax[2].set_xlabel("Wavenumber")
    ax[2].set_ylabel("Power Spectral Density")
    ax[2].legend()
    ax[2].grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()