import netCDF4
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from skimage import measure
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d
import matplotlib.image as mpimg
from matplotlib.colors import ListedColormap

# Get the 'hot' colormap with 256 colors
base_cmap = plt.cm.get_cmap('hot', 256)
cmap_data = base_cmap(np.linspace(0, 1, 256))

# Set the first color (value 0) to fully transparent
cmap_data[0, -1] = 0  # Last channel is alpha

# Create new colormap
transparent_cmap = ListedColormap(cmap_data)




drone_icon = mpimg.imread('drone.png')  # Should be a small transparent PNG
# === Load fire temperature data ===
nc = netCDF4.Dataset('FireTemp_field.nc', 'r')
temperature = np.squeeze(nc.variables['temperature'][:])  # Shape: (T, H, W)

# === Get ignition center from frame 0 ===
ignition_mask = temperature[0] >= 500
ignition_indices = np.argwhere(ignition_mask)
if ignition_indices.size == 0:
    raise ValueError("No fire found in frame 0.")
ignition_center = np.mean(ignition_indices, axis=0)  # (y, x)


# === Drone Control ===
kp = 0.06
ki = 0.00005
kd = 0.01
prevError = 0
integral = 0
max_step_size = 10
dt = 1

drone_x_pos, drone_y_pos = 0.0, 0.0
domainSize = temperature.shape[2]

# === PID Controller ===
def pid(kp, ki, kd, setpoint, currentPos):
    global prevError, integral
    error = np.linalg.norm(setpoint - currentPos)
    if error == 0:
        return np.array([0.0, 0.0])
    P = kp * error
    integral += ki * error * dt
    D = kd * (error - prevError) / dt
    PIDoutput = P + integral + D
    PIDoutput = np.clip(PIDoutput, -max_step_size, max_step_size)
    prevError = error
    dx = PIDoutput * (setpoint[0] - currentPos[0]) / error
    dy = PIDoutput * (setpoint[1] - currentPos[1]) / error
    return np.array([dx, dy])


# === Setup figure ===
fig, ax = plt.subplots(figsize=(8, 8))

# Optional smoothing buffer for tip location
tip_history = []

def smooth(val_list, N=5):
    if len(val_list) > N:
        val_list.pop(0)
    return np.mean(val_list, axis=0)

def update(frame):
    global drone_x_pos, drone_y_pos
    ax.clear()
    temp = temperature[frame]
    fire_mask = (temp >= 500).astype(float)
    ax.set_xlim(0, domainSize)
    ax.set_ylim(0, domainSize)
    ax.set_title(f"Frame {frame}")

    ax.imshow(temp, cmap="hot", interpolation='nearest', origin='upper')
    ax.set_title(f"Time: {frame} Seconds")


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
            ax.yaxis_inverted()
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

            length = 100
            x1 = x_tip_smooth - dx_unit * length / 2
            x2 = x_tip_smooth + dx_unit * length / 2
            y1 = y_tip_smooth - dy_unit * length / 2
            y2 = y_tip_smooth + dy_unit * length / 2
            ax.plot([x1, x2], [y1, y2], '-', linewidth=2, label='Tangent at Tip', color='red')
            
            # === Compute normal vector (perpendicular to tangent) ===
            # Normal is (-dy, dx) or (dy, -dx), pick one direction
            # === Offset along tangent direction (stay on the tangent line) ===
            desired_distance = -50  # negative to stay behind the fire tip, positive to lead ahead

            offset_x = x_tip_smooth + dx_unit * desired_distance
            offset_y = y_tip_smooth + dy_unit * desired_distance
            target_pos = np.array([offset_x, offset_y])
            drone_pos = np.array([drone_x_pos, drone_y_pos])
            velocity = pid(kp, ki, kd, target_pos, drone_pos)
            drone_x_pos += velocity[0] * dt
            drone_y_pos += velocity[1] * dt

            drone_x_pos = np.clip(drone_x_pos, 0, domainSize)
            drone_y_pos = np.clip(drone_y_pos, 0, domainSize)  

            # ax.plot(drone_x_pos, drone_y_pos, 'ro', label='Drone Position')
            icon_size = 10  # Size in data units

            ax.imshow(drone_icon,
                      extent=[drone_x_pos - icon_size/2, drone_x_pos + icon_size/2,
                              drone_y_pos - icon_size/2, drone_y_pos + icon_size/2],
                            zorder=10)
            ax.plot(*target_pos, 'x', label='Target Position')

        except Exception as e:
            print(f"Frame {frame}: Spline fitting failed — {e}")

    else:
        print(f"Frame {frame}: No fire contour found")
    # legend background white

    legend = ax.legend(loc='lower right', fontsize='small', frameon=True)
    legend.get_frame().set_facecolor('white')  # or any color like 'lightgray', '#f0f0f0'
    legend.get_frame().set_edgecolor('black')  # optional border color


# === Animate ===
ani = FuncAnimation(fig, update, frames=temperature.shape[0], interval=100)
#save andimation as gif
# ani.save('fire_animation.gif', writer='imagemagick', fps=60)
# Uncomment the following line to display the animation
# plt.show()
plt.show()
