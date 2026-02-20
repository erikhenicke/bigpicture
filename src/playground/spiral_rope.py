import numpy as np
import matplotlib.pyplot as plt

seq_len = 20
K = 100
phi = np.arange(K) * np.pi / K
p_x = np.broadcast_to(np.arange(-seq_len, seq_len).reshape(-1, 1), (2*seq_len, 2*seq_len))
p_y = np.broadcast_to(np.arange(-seq_len, seq_len).reshape(1, -1), (2*seq_len, 2*seq_len))
p = np.stack([p_x, p_y], axis=-1)

fig, axes = plt.subplots(10, 10, figsize=(15, 15))
axes = axes.flatten()

for k, phi_k in enumerate(phi):
    ax = axes[k]
    
    # Color based on the formula: x * cos(phi_k) + y * sin(phi_k)
    color_values = p[:, :, 0] * np.cos(phi_k) + p[:, :, 1] * np.sin(phi_k)
    color_values = (color_values - color_values.min()) / (color_values.max() - color_values.min())
    colors = plt.cm.Blues(color_values)
    ax.scatter(p[:, :, 0], p[:, :, 1], s=0.1, c=colors.reshape(-1, 4), label='Points', alpha=0.8)
    
    ax.set_xlim(-seq_len, seq_len)
    ax.set_ylim(-seq_len, seq_len)
    ax.set_aspect('equal')
    ax.set_title(f'φ = {(phi_k/np.pi):.2f}')
    if k == 0:
        ax.legend()

plt.tight_layout()
plt.show()
