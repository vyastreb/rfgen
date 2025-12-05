"""Tests for random field generators."""

import numpy as np
import pytest

from rfgen import selfaffine_field, matern_field


N = 256
k_low = 2 / N
k_high = 32 / N
Hurst = 0.5
seed = 42
rng = np.random.default_rng(seed)

field_1D = selfaffine_field(dim=1, N=N, Hurst=Hurst, k_low=k_low, k_high=k_high, rng=rng)
field_2D = selfaffine_field(dim=2, N=N, Hurst=Hurst, k_low=k_low, k_high=k_high, rng=rng)
field_3D = selfaffine_field(dim=3, N=N, Hurst=Hurst, k_low=k_low, k_high=k_high, rng=rng)

matern_1D = matern_field(dim=1, N=N, nu=2, correlation_length=0.1, k_high=k_high, rng=rng)
matern_2D = matern_field(dim=2, N=N, nu=2, correlation_length=0.1, k_high=k_high, rng=rng)
matern_3D = matern_field(dim=3, N=N, nu=2, correlation_length=0.1, k_high=k_high, rng=rng)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Create figure with custom column widths
fig = plt.figure(figsize=(15, 8))
gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 0.8], wspace=0.15, hspace=0.2)

# Self-affine fields
ax = fig.add_subplot(gs[0, 0])
ax.set_xlim(0, N-1)
ax.plot(field_1D, color="#b05030")
ax.set_title('Self-affine 1D Field')
ax.set_xticks([])
ax.set_yticks([])

ax = fig.add_subplot(gs[0, 1])
ax.imshow(field_2D, cmap='RdYlBu_r')
ax.set_title('Self-affine 2D Field')
ax.set_xticks([])
ax.set_yticks([])

ax = fig.add_subplot(gs[0, 2], projection='3d')
ax.set_box_aspect((1, 1, 1))
ax.axis('off')
norm = plt.Normalize(field_3D.min(), field_3D.max())
X, Y = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
# Face Z=0
ax.plot_surface(X, Y, np.zeros_like(X), facecolors=plt.cm.RdYlBu_r(norm(field_3D[:, :, 0])), shade=False, rstride=1, cstride=1, antialiased=False)
# Face Y=N-1
X_xz, Z_xz = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
ax.plot_surface(X_xz, np.full_like(X_xz, N-1), Z_xz, facecolors=plt.cm.RdYlBu_r(norm(field_3D[:, -1, :])), shade=False, rstride=1, cstride=1, antialiased=False)
# Face X=0
Y_yz, Z_yz = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
ax.plot_surface(np.zeros_like(Y_yz), Y_yz, Z_yz, facecolors=plt.cm.RdYlBu_r(norm(field_3D[0, :, :])), shade=False, rstride=1, cstride=1, antialiased=False)
# Edges
kwargs = {'color': '0.2', 'linewidth': 0.5, 'zorder': 10}
ax.plot([0, N-1], [N-1, N-1], [0, 0], **kwargs)
ax.plot([0, 0], [N-1, 0], [0, 0], **kwargs)
ax.plot([0, 0], [N-1, N-1], [0, N-1], **kwargs)
ax.plot([N-1, N-1, 0, 0, 0, N-1, N-1], 
        [N-1, 0, 0, 0, N-1, N-1, N-1], 
        [0, 0, 0, N-1, N-1, N-1, 0], **kwargs)

zoom_factor = 0.95
center = (N-1) / 2
range_val = (N-1) * zoom_factor / 2
ax.set_xlim([center - range_val, center + range_val])
ax.set_ylim([center - range_val, center + range_val])
ax.set_zlim([center - range_val, center + range_val])
ax.set_title('Self-affine 3D Field')

# Matern fields
ax = fig.add_subplot(gs[1, 0])
ax.set_xlim(0, N-1)
ax.plot(matern_1D, color="#305090")
ax.set_title('Matern 1D Field')
ax.set_xticks([])
ax.set_yticks([])

ax = fig.add_subplot(gs[1, 1])
ax.imshow(matern_2D, cmap='jet')
ax.set_title('Matern 2D Field')
ax.set_xticks([])
ax.set_yticks([])

ax = fig.add_subplot(gs[1, 2], projection='3d')
ax.set_box_aspect((1, 1, 1))
ax.axis('off')
norm_m = plt.Normalize(matern_3D.min(), matern_3D.max())
ax.plot_surface(X, Y, np.zeros_like(X), facecolors=plt.cm.jet(norm_m(matern_3D[:, :, 0])), shade=False, rstride=1, cstride=1, antialiased=False)
ax.plot_surface(X_xz, np.full_like(X_xz, N-1), Z_xz, facecolors=plt.cm.jet(norm_m(matern_3D[:, -1, :])), shade=False, rstride=1, cstride=1, antialiased=False)
ax.plot_surface(np.zeros_like(Y_yz), Y_yz, Z_yz, facecolors=plt.cm.jet(norm_m(matern_3D[0, :, :])), shade=False, rstride=1, cstride=1, antialiased=False)
ax.plot([0, N-1], [N-1, N-1], [0, 0], **kwargs)
ax.plot([0, 0], [N-1, 0], [0, 0], **kwargs)
ax.plot([0, 0], [N-1, N-1], [0, N-1], **kwargs)
ax.plot([N-1, N-1, 0, 0, 0, N-1, N-1], 
        [N-1, 0, 0, 0, N-1, N-1, N-1], 
        [0, 0, 0, N-1, N-1, N-1, 0], **kwargs)

ax.set_xlim([center - range_val, center + range_val])
ax.set_ylim([center - range_val, center + range_val])
ax.set_zlim([center - range_val, center + range_val])
ax.set_title('Matern 3D Field')

fig.tight_layout()
fig.savefig('random_fields.png', bbox_inches='tight', pad_inches=0, dpi=150)
plt.show()
