import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import netCDF4

# === Fire Data ===
nc = netCDF4.Dataset('FireTemp_field.nc', 'r')
temperature = nc.variables['temperature'][:]
t_simDur = temperature.shape[0]

ignition_mask = temperature[0] >= 500
ignition_indices = np.argwhere(ignition_mask)
if ignition_indices.size == 0:
    raise ValueError("No fire found in frame 0 to determine ignition center.")
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

# === Animation ===
fig, ax = plt.subplots(figsize=(8, 8))

def update(frame):
    global drone_x_pos, drone_y_pos
    ax.clear()
    temp = temperature[frame]
    fire_mask = temp >= 500

    ax.set_xlim(0, domainSize)
    ax.set_ylim(0, domainSize)
    ax.set_title(f"Frame {frame}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    if np.any(fire_mask):
        fire_indices = np.argwhere(fire_mask)
        distances = np.linalg.norm(fire_indices - ignition_center, axis=1)
        tip_index = fire_indices[np.argmax(distances)]
        y_tip, x_tip = int(tip_index[0]), int(tip_index[1])
        tip_temp = temp[y_tip, x_tip]

        # Target is slightly offset from fire tip (optional)
        target_pos = np.array([x_tip + 10, y_tip])
        drone_pos = np.array([drone_x_pos, drone_y_pos])
        velocity = pid(kp, ki, kd, target_pos, drone_pos)
        drone_x_pos += velocity[0] * dt
        drone_y_pos += velocity[1] * dt

        drone_x_pos = np.clip(drone_x_pos, 0, domainSize)
        drone_y_pos = np.clip(drone_y_pos, 0, domainSize)

        # Plot fire
        im = ax.imshow(temp, cmap='hot', interpolation='nearest')
        ax.scatter(x_tip, y_tip, color='red', s=40, label='Fire Tip')
        ax.scatter(*target_pos, color='black', marker='x', s=40, label='Target')

        # Plot drone
        ax.plot(drone_x_pos, drone_y_pos, 'bo', markersize=7, label='Drone')
        # ax.plot(0, domainSize, 'bo', markersize=0, label=f'Drone Dist: {np.linalg.norm(drone_pos - target_pos):.1f} m')

        # if hasattr(update, "colorbar"):
        #     update.colorbar.remove()
        # update.colorbar = fig.colorbar(im, ax=ax, label='Temp (K)')

    else:
        ax.imshow(temp, cmap='hot', interpolation='nearest')
        ax.set_title(f"Frame {frame} (No Fire)")

    ax.legend(loc='lower right', fontsize='small', frameon=False)

ani = FuncAnimation(fig, update, frames=t_simDur, interval=150)
# savve as gif
ani.save('fire_drone.gif', writer='imagemagick', fps=60)
plt.show()
