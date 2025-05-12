import netCDF4
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Load the netCDF file
nc = netCDF4.Dataset('FireTemp_field.nc', 'r')
temperature = nc.variables['temperature'][:]

# Determine fire origin from first frame
ignition_mask = temperature[0] >= 500
ignition_indices = np.argwhere(ignition_mask)

if ignition_indices.size == 0:
    raise ValueError("No fire found in frame 0 to determine ignition center.")

ignition_center = np.mean(ignition_indices, axis=0)  # (y, x)

# Storage for fire tip locations
max_values = []

# Setup figure
fig, ax = plt.subplots(figsize=(8, 8))

def update(frame):
    ax.clear()

    temp = temperature[frame]
    fire_mask = temp >= 500

    if np.any(fire_mask):
        fire_indices = np.argwhere(fire_mask)  # shape: (N, 2)

        # Compute distance from ignition point
        distances = np.linalg.norm(fire_indices - ignition_center, axis=1)
        tip_index = fire_indices[np.argmax(distances)]  # shape: (2,)

        y_tip, x_tip = int(tip_index[0]), int(tip_index[1])
        tip_temp = float(temp[y_tip, x_tip])

        max_values.append((frame, x_tip, y_tip, tip_temp))

        im = ax.imshow(temp, cmap='hot', interpolation='nearest')
        ax.scatter(x_tip, y_tip, color='cyan', s=30, label='Fire Tip')
        ax.set_title(f"Frame {frame}")
        # fig.colorbar(im, ax=ax, label='Temperature (K)')
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.legend()
    else:
        ax.imshow(temp, cmap='hot', interpolation='nearest')
        ax.set_title(f"Frame {frame} (No Fire)")
        # fig.colorbar(ax.images[0], ax=ax, label='Temperature (K)')
        print(f"Frame {frame}: No fire detected")

# Run animation

ani = FuncAnimation(fig, update, frames=temperature.shape[0], interval=100)
plt.show()
