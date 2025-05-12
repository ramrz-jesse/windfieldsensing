import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

droneCount = 1  # Number of drones to simulate
t_tod = 3 # Time of deployment in seconds
t_simDur = 120 # Duration of simulation in seconds

#PID control variables
kp = 0.06  # Proportional gain
ki = 0.00005  # Integral gain
kd = 0.01
prevError = 0
prevIntegral = 0
integral = 0


# Create grid based on domain size
domainSize = 500
x = np.linspace(0, domainSize, (domainSize * 10) + 2)
y = np.linspace(0, domainSize, (domainSize * 10) + 2)
X, Y = np.meshgrid(x, y)



 # Initialize particle position
particle_x_pos = np.random.uniform(domainSize / 2, domainSize)
particle_y_pos = np.random.uniform(domainSize / 2, domainSize)

# Initialize drone position
drone_x_pos = 0
drone_y_pos = 0


# Maximum step size for the drone's movement
max_step_size = 10  # Adjust this value as needed for realism
dt = 1  # Time step for simulation
# PID control with step size limit
def pid(kp, ki, kd, setpoint, currentPos):
    global prevError, integral
    error = np.linalg.norm(setpoint - currentPos)
    # error = setpoint - currentPos
    # de = error - prevError  # Change in error
    P = kp * error  # Proportional termq
    integral = integral + ki*error*dt # Integral of error
    d = kd * (error - prevError)/dt
   

    PIDoutput = P + integral + d  # PID output
    # Limit the output to the maximum step size
    PIDoutput = np.clip(PIDoutput, -max_step_size, max_step_size)
    prevError = error  # Store current error for next iteration
    

    # output velocity for each axis
    dx = PIDoutput * (setpoint[0] - currentPos[0]) / error
    dy = PIDoutput * (setpoint[1] - currentPos[1]) / error

    return np.array([dx, dy])




# Update function for animation
def update(frame):
    global particle_x_pos, particle_y_pos, drone_x_pos, drone_y_pos

    plt.cla()  # Clear the axes instead of the figure to retain settings
    plt.title("Simulated Wind Field with Drone Particle Tracking")
    plt.xlabel("X - Axis")
    plt.ylabel("Y - Axis")

    plt.xlim(0, domainSize)
    plt.ylim(0, domainSize)
    


    if frame >= t_tod:        
        droneDist = np.sqrt((drone_x_pos - particle_x_pos)**2 + (drone_y_pos - particle_y_pos+100)**2) 
        plt.plot(drone_x_pos, drone_y_pos, 'bo', markersize=7 , label=f'Drone')
        plt.plot(0, domainSize, 'bo', markersize=0, label = f'Drone Dist.: {droneDist:.2f} m')
        particle_x_pos += np.random.choice([-1, 1])
        particle_y_pos += np.random.choice([-1, 1])
        # Close into the particle's position using PID control
        target_pos = np.array([particle_x_pos - 10, particle_y_pos + 10]) 
        velocity = pid(kp, ki,kd, target_pos, np.array([drone_x_pos, drone_y_pos]))
        drone_x_pos += velocity[0] * dt
        drone_y_pos += velocity[1] * dt
        print(f"Drone Velocity: {velocity[0]:.2f}, {velocity[1]:.2f}")
        # Update drone position
        # print drone velocity for each axis
    
    # Keep particle & drone within bounds
    particle_x_pos = np.clip(particle_x_pos, 0, domainSize)
    particle_y_pos = np.clip(particle_y_pos, 0, domainSize)
    drone_x_pos = np.clip(drone_x_pos, 0, domainSize)
    drone_y_pos = np.clip(drone_y_pos, 0, domainSize)

    # Update particle plot
    plt.plot(particle_x_pos, particle_y_pos, 'ro', markersize=5, label=f'Particle')
    plt.plot(particle_x_pos - 10, particle_y_pos + 10, 'rx', markersize=5, label=f'target')
    plt.plot(0, domainSize, 'ro', markersize=0, label = f'Time: {frame:.2f} s')
    
    plt.legend(loc='lower right', fontsize='small', frameon=False)
# Create animation
fig = plt.figure(figsize=(8, 8))
ani = FuncAnimation(fig, update, frames = t_simDur, interval = 250, repeat = True) # Set interval to 1000 for real time

# Uncomment following lines to save the animation as a gif
# writer = animation.PillowWriter(fps = 2, metadata=dict(artist='Me'), bitrate=1800)
# ani.save('particleTrack.gif', writer=writer)

plt.show()



# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# import netCDF4

# # === Fire Data ===
# nc = netCDF4.Dataset('FireTemp_field.nc', 'r')
# temperature = nc.variables['temperature'][:]
# t_simDur = temperature.shape[0]

# ignition_mask = temperature[0] >= 500
# ignition_indices = np.argwhere(ignition_mask)
# if ignition_indices.size == 0:
#     raise ValueError("No fire found in frame 0 to determine ignition center.")
# ignition_center = np.mean(ignition_indices, axis=0)  # (y, x)

# # === Drone Control ===
# kp = 0.06
# ki = 0.00005
# kd = 0.01
# prevError = 0
# integral = 0
# max_step_size = 10
# dt = 1

# drone_x_pos, drone_y_pos = 0.0, 0.0
# domainSize = temperature.shape[2]

# # === PID Controller ===
# def pid(kp, ki, kd, setpoint, currentPos):
#     global prevError, integral
#     error = np.linalg.norm(setpoint - currentPos)
#     if error == 0:
#         return np.array([0.0, 0.0])
#     P = kp * error
#     integral += ki * error * dt
#     D = kd * (error - prevError) / dt
#     PIDoutput = P + integral + D
#     PIDoutput = np.clip(PIDoutput, -max_step_size, max_step_size)
#     prevError = error
#     dx = PIDoutput * (setpoint[0] - currentPos[0]) / error
#     dy = PIDoutput * (setpoint[1] - currentPos[1]) / error
#     return np.array([dx, dy])

# # === Animation ===
# fig, ax = plt.subplots(figsize=(8, 8))

# def update(frame):
#     global drone_x_pos, drone_y_pos
#     ax.clear()
#     temp = temperature[frame]
#     fire_mask = temp >= 500

#     ax.set_xlim(0, domainSize)
#     ax.set_ylim(0, domainSize)
#     ax.set_title(f"Frame {frame}")
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")

#     if np.any(fire_mask):
#         fire_indices = np.argwhere(fire_mask)
#         distances = np.linalg.norm(fire_indices - ignition_center, axis=1)
#         tip_index = fire_indices[np.argmax(distances)]
#         y_tip, x_tip = int(tip_index[0]), int(tip_index[1])
#         tip_temp = temp[y_tip, x_tip]

#         # Target is slightly offset from fire tip (optional)
#         target_pos = np.array([x_tip + 10, y_tip])
#         drone_pos = np.array([drone_x_pos, drone_y_pos])
#         velocity = pid(kp, ki, kd, target_pos, drone_pos)
#         drone_x_pos += velocity[0] * dt
#         drone_y_pos += velocity[1] * dt

#         drone_x_pos = np.clip(drone_x_pos, 0, domainSize)
#         drone_y_pos = np.clip(drone_y_pos, 0, domainSize)

#         # Plot fire
#         im = ax.imshow(temp, cmap='hot', interpolation='nearest')
#         ax.scatter(x_tip, y_tip, color='red', s=40, label='Fire Tip')
#         ax.scatter(*target_pos, color='black', marker='x', s=40, label='Target')

#         # Plot drone
#         ax.plot(drone_x_pos, drone_y_pos, 'bo', markersize=7, label='Drone')
#         # ax.plot(0, domainSize, 'bo', markersize=0, label=f'Drone Dist: {np.linalg.norm(drone_pos - target_pos):.1f} m')

#         # if hasattr(update, "colorbar"):
#         #     update.colorbar.remove()
#         # update.colorbar = fig.colorbar(im, ax=ax, label='Temp (K)')

#     else:
#         ax.imshow(temp, cmap='hot', interpolation='nearest')
#         ax.set_title(f"Frame {frame} (No Fire)")

#     ax.legend(loc='lower right', fontsize='small', frameon=False)

# ani = FuncAnimation(fig, update, frames=t_simDur, interval=150)
# # savve as gif
# ani.save('fire_drone.gif', writer='imagemagick', fps=60)
# plt.show()
