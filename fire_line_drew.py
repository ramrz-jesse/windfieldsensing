import netCDF4
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from skimage import measure
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d

# === Load fire temperature data ===
nc = netCDF4.Dataset('FireTemp_field.nc', 'r')
temperature = np.squeeze(nc.variables['temperature'][:])  # Shape: (T, H, W)

# === Get ignition center from frame 0 ===
ignition_mask = temperature[0] >= 500
ignition_indices = np.argwhere(ignition_mask)
if ignition_indices.size == 0:
    raise ValueError("No fire found in frame 0.")
ignition_center = np.mean(ignition_indices, axis=0)  # (y, x)

# === Setup figure ===
fig, ax = plt.subplots(figsize=(8, 8))

# Optional smoothing buffer for tip location
tip_history = []

def smooth(val_list, N=5):
    if len(val_list) > N:
        val_list.pop(0)
    return np.mean(val_list, axis=0)

def update(frame):
    ax.clear()
    temp = temperature[frame]
    fire_mask = (temp >= 500).astype(float)

    ax.imshow(temp, cmap='hot', interpolation='nearest', origin='upper')
    ax.set_title(f"Frame {frame}")
    ax.set_xlabel("X-axis (columns)")
    ax.set_ylabel("Y-axis (rows)")

    contours = measure.find_contours(fire_mask, level=0.5)

    if contours:
        front = max(contours, key=len)
        if len(front) < 10:
            print(f"Frame {frame}: Too few contour points")
            return

        y_vals = front[:, 0]
        x_vals = front[:, 1]

        # === Smooth contour coordinates ===
        x_smooth = gaussian_filter1d(x_vals, sigma=2)
        y_smooth = gaussian_filter1d(y_vals, sigma=2)

        try:
            # === Fit parametric spline to smoothed contour ===
            tck, u = splprep([x_smooth, y_smooth], s=10)
            u_fine = np.linspace(0, 1, 500)
            x_spline, y_spline = splev(u_fine, tck)

            # Plot the spline
            ax.plot(x_spline, y_spline, 'g--', label='Fire Front Fit')

            # === Fire tip (farthest from ignition) ===
            ignition_center_xy = ignition_center[::-1]  # (x, y)
            distances = np.linalg.norm(np.vstack([x_spline, y_spline]).T - ignition_center_xy, axis=1)
            tip_idx = np.argmax(distances)
            x_tip, y_tip = x_spline[tip_idx], y_spline[tip_idx]

            # Optionally smooth tip
            tip_history.append([x_tip, y_tip])
            x_tip_smooth, y_tip_smooth = smooth(tip_history)

            ax.plot(x_tip_smooth, y_tip_smooth, 'co', label='Fire Tip')

            # === Tangent line at tip ===
            dx, dy = splev(u_fine[tip_idx], tck, der=1)
            mag = np.hypot(dx, dy)
            dx_unit, dy_unit = dx / mag, dy / mag

            length = 40
            x1 = x_tip_smooth - dx_unit * length / 2
            x2 = x_tip_smooth + dx_unit * length / 2
            y1 = y_tip_smooth - dy_unit * length / 2
            y2 = y_tip_smooth + dy_unit * length / 2
            ax.plot([x1, x2], [y1, y2], 'b-', linewidth=2, label='Tangent at Tip', color='red')

            angle_deg = np.degrees(np.arctan2(-dy, dx))
            print(f"Frame {frame}: Tangent angle = {angle_deg:.2f}°")

        except Exception as e:
            print(f"Frame {frame}: Spline fitting failed — {e}")

    else:
        print(f"Frame {frame}: No fire contour found")
    # legend background white

    ax.legend(loc='lower right', fontsize='small', frameon=False, )



# === Animate ===
ani = FuncAnimation(fig, update, frames=temperature.shape[0], interval=100)
#save andimation as gif
ani.save('fire_animation.gif', writer='imagemagick', fps=60)
# Uncomment the following line to display the animation
# plt.show()
plt.show()
