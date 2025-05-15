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
cmap_data[0, -1] = 0  # Make value-0 fully transparent
transparent_cmap = ListedColormap(cmap_data)
rgb_img = np.load('rgb_image.npy')  # Load the RGB image
# rotate the image 90 degrees
# rgb_img = np.rot90(rgb_img, k=3)  # Rotate the image 90 degrees counter-clockwise
# flip the image vertically
# rgb_img = np.flipud(rgb_img)  # Flip the image vertically
drone_icon = mpimg.imread('drone.png')  # small transparent PNG

# === Load fire temperature data ===
nc = netCDF4.Dataset('FireTemp_field-noFireBreak-new.nc', 'r')
temperature = np.squeeze(nc.variables['temperature'][:])  # Shape: (T, H, W)

# flip x and y

temperature = np.rot90(temperature, k=3, axes=(1, 2))
# flip the image horizontally
# mirror the temperature data
temperature = np.flip(temperature, axis=2)  # Flip the temperature data horizontally



# === Get ignition center from frame 0 ===
ignition_mask = temperature[0] >= 500
ignition_indices = np.argwhere(ignition_mask)
if ignition_indices.size == 0:
    raise ValueError("No fire found in frame 0.")
ignition_center = np.mean(ignition_indices, axis=0)  # (y, x)

# === Drone PID parameters ===
kp, ki, kd = 0.06, 0.00005, 0.01
prevError = 0
integral = 0
max_step_size = 10
dt = 1
drone_x_pos, drone_y_pos = 0.0, 0.0
domainSize = temperature.shape[2]

# === Storage for past fire fronts ===
past_fire_fronts = []
past_tangents = []
past_drones = []

def pid(kp, ki, kd, setpoint, currentPos):
    global prevError, integral
    error = np.linalg.norm(setpoint - currentPos)
    if error == 0:
        return np.array([0.0, 0.0])
    P = kp * error
    integral += ki * error * dt
    D = kd * (error - prevError) / dt
    PIDoutput = np.clip(P + integral + D, -max_step_size, max_step_size)
    prevError = error
    dx = PIDoutput * (setpoint[0] - currentPos[0]) / error
    dy = PIDoutput * (setpoint[1] - currentPos[1]) / error
    return np.array([dx, dy])

# === Setup figure ===
fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
tip_history = []

def smooth(val_list, N=5):
    if len(val_list) > N:
        val_list.pop(0)
    return np.mean(val_list, axis=0)

def update(frame):
    global drone_x_pos, drone_y_pos
    ax.clear()
    temp = temperature[frame]
    fire_mask = (temp >= 300).astype(float)
    meter_max = 500
    tick_meters = np.arange(0, meter_max + 1, 100)  # [0, 100, 200, ..., 500]
    tick_positions = tick_meters * (domainSize / meter_max)  # scaled to data

    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(m) for m in tick_meters], fontsize=18)

    ax.set_yticks(tick_positions)
    ax.set_yticklabels([str(m) for m in tick_meters], fontsize=18)

    ax.set_xlim(0, domainSize)
    ax.set_ylim(0, domainSize)

    ax.set_xlabel("X (m)", fontsize=18)
    ax.set_ylabel("Y (m)", fontsize=18)
    ax.imshow(rgb_img, extent=[0, domainSize, 0, domainSize], zorder=1)
    # ax.imshow(temp, cmap=transparent_cmap, interpolation='nearest', origin='upper', alpha=0.5, zorder=2)

    contours = measure.find_contours(fire_mask, level=0.5)
    if not contours:
        print(f"Frame {frame}: No fire contour found")
        return

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

        # Store every 100th frame (for fire front trails)
        if frame % 50 == 0:
            past_fire_fronts.append((x_spline, y_spline))

        # Plot stored fire fronts
        for xf, yf in past_fire_fronts:
            ax.plot(xf, yf, 'r-', alpha=0.7, linewidth=3)

        # Highlight current fit
        if frame % 100 == 0:
            ax.plot(x_vals, y_vals, 'g--', label='Fire Front Fit')

        # Tip tracking
        ignition_center_xy = ignition_center[::-1]
        distances = np.linalg.norm(np.vstack([x_spline, y_spline]).T - ignition_center_xy, axis=1)
        tip_idx = np.argmax(distances)
        x_tip, y_tip = x_spline[tip_idx], y_spline[tip_idx]
        tip_history.append([x_tip, y_tip])
        x_tip_smooth, y_tip_smooth = smooth(tip_history)

        # Tangent at tip
        dx, dy = splev(u_fine[tip_idx], tck, der=1)
        mag = np.hypot(dx, dy)
        dx_unit, dy_unit = dx / mag, dy / mag
        length = 30
        x1 = x_tip_smooth - dx_unit * length / 2
        x2 = x_tip_smooth + dx_unit * length / 2
        y1 = y_tip_smooth - dy_unit * length / 2
        y2 = y_tip_smooth + dy_unit * length / 2
        # ax.plot([x1, x2], [y1, y2], '-', linewidth=2, label='Tangent at Tip', color='orange')

        # Drone motion
        desired_distance = 15
        offset_x = x_tip_smooth + dx_unit * desired_distance
        offset_y = y_tip_smooth + dy_unit * desired_distance
        target_pos = np.array([offset_x, offset_y])
        velocity = pid(kp, ki, kd, target_pos, np.array([drone_x_pos, drone_y_pos]))
        drone_x_pos += velocity[0] * dt
        drone_y_pos += velocity[1] * dt
        drone_x_pos = np.clip(drone_x_pos, 0, domainSize)
        drone_y_pos = np.clip(drone_y_pos, 0, domainSize)

        # Store drone + tangent every 200 frames after 300
        if frame >= 400 and frame % 200 == 0:
            past_drones.append((drone_x_pos, drone_y_pos))
            past_tangents.append(((x1, x2), (y1, y2)))

        # Plot drone icon at current position
        icon_size = 3
        # ax.imshow(drone_icon,
        #           extent=[drone_x_pos - icon_size / 2, drone_x_pos + icon_size / 2,
        #                   drone_y_pos - icon_size / 2, drone_y_pos + icon_size / 2],
        #         #   zorder=10)

        # Plot stored historical drone positions and tangent lines
        for (dx_hist, dy_hist), ((xt1, xt2), (yt1, yt2)) in zip(past_drones, past_tangents):
            ax.imshow(drone_icon,
                      extent=[dx_hist - icon_size / 2, dx_hist + icon_size / 2,
                              dy_hist - icon_size / 2, dy_hist + icon_size / 2],
                      zorder=5, alpha=0.9)
            ax.plot([xt1, xt2], [yt1, yt2], '--', linewidth=3, color='orange', alpha=0.9)

    except Exception as e:
        print(f"Frame {frame}: Spline fitting failed — {e}")

    # Legend
    # legend = ax.legend(loc='lower right', fontsize='small', frameon=True)
    # legend.get_frame().set_facecolor('white')
    # legend.get_frame().set_edgecolor('black')

# Animate
ani = FuncAnimation(fig, update, frames=temperature.shape[0], interval=50, repeat=False)
plt.show()
# === Save final frame ===
fig_final, ax_final = plt.subplots(figsize=(10, 10), dpi=100)

# Axis scaling and labeling
meter_max = 500
tick_meters = np.arange(0, meter_max + 1, 100)
tick_positions = tick_meters * (domainSize / meter_max)

ax_final.set_xticks(tick_positions)
ax_final.set_xticklabels([str(m) for m in tick_meters], fontsize=18)
ax_final.set_yticks(tick_positions)
ax_final.set_yticklabels([str(m) for m in tick_meters], fontsize=18)

ax_final.set_xlim(0, domainSize)
ax_final.set_ylim(0, domainSize)
ax_final.set_xlabel("X (m)", fontsize=18)
ax_final.set_ylabel("Y (m)", fontsize=18)

# RGB background
ax_final.imshow(rgb_img, extent=[0, domainSize, 0, domainSize], zorder=1)

# Plot stored fire fronts
for xf, yf in past_fire_fronts:
    ax_final.plot(xf, yf, 'r-', alpha=0.7, linewidth=3)

# Plot stored drone positions and tangents
icon_size = 3
for (dx_hist, dy_hist), ((xt1, xt2), (yt1, yt2)) in zip(past_drones, past_tangents):
    ax_final.imshow(drone_icon,
                    extent=[dx_hist - icon_size / 2, dx_hist + icon_size / 2,
                            dy_hist - icon_size / 2, dy_hist + icon_size / 2],
                    zorder=5, alpha=0.9)
    ax_final.plot([xt1, xt2], [yt1, yt2], '--', linewidth=3, color='orange', alpha=0.9)

# # Plot current (final) drone location
# ax_final.imshow(drone_icon,
#                 extent=[drone_x_pos - icon_size / 2, drone_x_pos + icon_size / 2,
#                         drone_y_pos - icon_size / 2, drone_y_pos + icon_size / 2],
#                 zorder=10)

# Save
plt.tight_layout()
plt.savefig("final_firefront_with_drone.png", dpi=300)
plt.close(fig_final)
print("Saved final frame to 'final_firefront_with_drone.png'")


# # === Save final frame as static image ===
# fig_final, ax_final = plt.subplots(figsize=(10, 10), dpi=100)

# def render_final_frame(frame):
#     global drone_x_pos, drone_y_pos

#     # Exact same content as update() — but just for the final frame and using ax_final
#     temp = temperature[frame]
#     fire_mask = (temp >= 300).astype(float)
#     meter_max = 500
#     tick_meters = np.arange(0, meter_max + 1, 100)
#     tick_positions = tick_meters * (domainSize / meter_max)

#     ax_final.set_xticks(tick_positions)
#     ax_final.set_xticklabels([str(m) for m in tick_meters], fontsize=16)
#     ax_final.set_yticks(tick_positions)
#     ax_final.set_yticklabels([str(m) for m in tick_meters], fontsize=16)

#     ax_final.set_xlim(0, domainSize)
#     ax_final.set_ylim(0, domainSize)
#     ax_final.set_xlabel("X (m)", fontsize=12)
#     ax_final.set_ylabel("Y (m)", fontsize=12)
#     ax_final.imshow(rgb_img, extent=[0, domainSize, 0, domainSize], zorder=1)

#     for xf, yf in past_fire_fronts:
#         ax_final.plot(xf, yf, 'r-', alpha=0.7)

#     for (dx_hist, dy_hist), ((xt1, xt2), (yt1, yt2)) in zip(past_drones, past_tangents):
#         ax_final.imshow(drone_icon,
#                         extent=[dx_hist - icon_size / 2, dx_hist + icon_size / 2,
#                                 dy_hist - icon_size / 2, dy_hist + icon_size / 2],
#                         zorder=5, alpha=0.9)
#         ax_final.plot([xt1, xt2], [yt1, yt2], '--', linewidth=1.5, color='orange', alpha=0.9)

#     icon_size = 3
#     ax_final.imshow(drone_icon,
#                     extent=[drone_x_pos - icon_size / 2, drone_x_pos + icon_size / 2,
#                             drone_y_pos - icon_size / 2, drone_y_pos + icon_size / 2],
#                     zorder=10)

# # Pick last frame from animation
# final_frame_idx = temperature.shape[0] - 1
# render_final_frame(final_frame_idx)

# # Save to file
# plt.tight_layout()
# plt.savefig("final_firefront_with_drone.png", dpi=300)
# print("Saved final frame to 'final_firefront_with_drone.png'")
