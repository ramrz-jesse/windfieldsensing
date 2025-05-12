import netCDF4
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from skimage import measure
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d

# === Load fire temperature data ===
nc = netCDF4.Dataset('data/FireTemp_field.nc', 'r')
temperature = np.squeeze(nc.variables['temperature'][:])  # Shape: (T, H, W)

# === Get ignition center from frame 0 ===
ignition_mask = temperature[0] >= 500
ignition_indices = np.argwhere(ignition_mask)
if ignition_indices.size == 0:
    raise ValueError("No fire found in frame 0.")
ignition_center = np.mean(ignition_indices, axis=0)  # (y, x)

# === Simulation parameters ===
t_tod = 3  # Time of deployment (seconds)
tangent_offset = 20  # Distance along tangent (meters)
domain_size = temperature.shape[2]  # Assuming square grid

drone_x_pos, drone_y_pos = 0.0, 250.0  # Initialize drone at (0,0)
prev_error = 0
integral = 0
max_step_size = 10  # Drone speed limit (m per frame)
dt = 1  # Time step (assume 1 second per frame)

# Track once determined
drone_side_decided = False
drone_offset_direction = 1  # +1 or -1

# === PID controller ===
def pid(kp, ki, kd, setpoint, current_pos):
    global prev_error, integral
    error_vec = setpoint - current_pos
    error_mag = np.linalg.norm(error_vec)
    if error_mag == 0:
        return np.array([0.0, 0.0])
    P = kp * error_mag
    integral += ki * error_mag * dt
    D = kd * (error_mag - prev_error) / dt
    output = P + integral + D
    output = np.clip(output, -max_step_size, max_step_size)
    prev_error = error_mag
    dx = output * (error_vec[0] / error_mag)
    dy = output * (error_vec[1] / error_mag)
    return np.array([dx, dy])

# PID gains
kp = 0.06
ki = 0.00005
kd = 0.01

# === Visualization ===
fig, ax = plt.subplots(figsize=(8, 8))

# Tip smoothing buffer
tip_history = []

def smooth(val_list, N=5):
    if len(val_list) > N:
        val_list.pop(0)
    return np.mean(val_list, axis=0)

# === Animation update ===
def update(frame):
    global drone_x_pos, drone_y_pos, drone_side_decided, drone_offset_direction

    ax.clear()
    temp = temperature[frame]
    fire_mask = (temp >= 500).astype(float)

    ax.imshow(temp, cmap='hot', interpolation='nearest', origin='upper')
    ax.set_xlim(0, domain_size)
    ax.set_ylim(domain_size, 0)
    ax.set_title(f"Frame {frame}")
    ax.set_xlabel("X-axis (columns)")
    ax.set_ylabel("Y-axis (rows)")
    ax.invert_yaxis()

    contours = measure.find_contours(fire_mask, level=0.5)

    if contours:
        front = max(contours, key=len)
        if len(front) < 10:
            print(f"Frame {frame}: Too few contour points")
            return

        y_vals, x_vals = front[:, 0], front[:, 1]

        x_smooth = gaussian_filter1d(x_vals, sigma=2)
        y_smooth = gaussian_filter1d(y_vals, sigma=2)

        try:
            tck, u = splprep([x_smooth, y_smooth], s=10)
            u_fine = np.linspace(0, 1, 500)
            x_spline, y_spline = splev(u_fine, tck)

            ax.plot(x_spline, y_spline, 'g--', label='Fire Front Fit')

            ignition_center_xy = ignition_center[::-1]  # (x, y)
            distances = np.linalg.norm(np.vstack([x_spline, y_spline]).T - ignition_center_xy, axis=1)
            tip_idx = np.argmax(distances)
            x_tip, y_tip = x_spline[tip_idx], y_spline[tip_idx]

            tip_history.append([x_tip, y_tip])
            x_tip_smooth, y_tip_smooth = smooth(tip_history)

            ax.plot(x_tip_smooth, y_tip_smooth, 'co', label='Fire Tip')

            # === Tangent Calculation ===
            dx, dy = splev(u_fine[tip_idx], tck, der=1)
            mag = np.hypot(dx, dy)
            dx_unit, dy_unit = dx / mag, dy / mag

            # Draw tangent line
            length = 40
            x1 = x_tip_smooth - dx_unit * length / 2
            x2 = x_tip_smooth + dx_unit * length / 2
            y1 = y_tip_smooth - dy_unit * length / 2
            y2 = y_tip_smooth + dy_unit * length / 2
            ax.plot([x1, x2], [y1, y2], linewidth=2, label='Tangent at Tip', color='red')

            if frame >= t_tod:
                # First deployment: decide drone side
                if not drone_side_decided:
                    drone_pos = np.array([drone_x_pos, drone_y_pos])
                    tip_pos = np.array([x_tip_smooth, y_tip_smooth])
                    tangent_vec = np.array([dx_unit, dy_unit])
                    drone_vec = drone_pos - tip_pos
                    cross = np.cross(tangent_vec, drone_vec)
                    drone_offset_direction = -1 if cross >= 0 else 1
                    drone_side_decided = True
                    print(f"Drone side decided: {'ahead' if drone_offset_direction == 1 else 'behind'} the tangent")

                # Offset along tangent
                target_pos = np.array([x_tip_smooth, y_tip_smooth]) + drone_offset_direction * np.array([dx_unit, dy_unit]) * tangent_offset

                # PID move drone
                current_pos = np.array([drone_x_pos, drone_y_pos])
                velocity = pid(kp, ki, kd, target_pos, current_pos)
                drone_x_pos += velocity[0] * dt
                drone_y_pos += velocity[1] * dt

                drone_x_pos = np.clip(drone_x_pos, 0, domain_size)
                drone_y_pos = np.clip(drone_y_pos, 0, domain_size)

                ax.plot(drone_x_pos, drone_y_pos, 'bo', markersize=7, label='Drone')
                ax.plot(target_pos[0], target_pos[1], 'rx', markersize=7, label='Target')

                ax.plot([x_tip_smooth, target_pos[0]], [y_tip_smooth, target_pos[1]], 'r:', label='Offset Path')

            angle_deg = np.degrees(np.arctan2(-dy, dx))
            print(f"Frame {frame}: Tangent angle = {angle_deg:.2f}°")

        except Exception as e:
            print(f"Frame {frame}: Spline fitting failed — {e}")
            return
    else:
        print(f"Frame {frame}: No fire contour found")

    ax.legend(loc='lower right', fontsize='small', frameon=False)

# === Run animation ===
ani = FuncAnimation(fig, update, frames=temperature.shape[0], interval=150)
ani.save('fire_drone.gif', writer='imagemagick', fps=60)
plt.show()
