import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

droneCount = 1  # Number of drones to simulate
t_tod = 5 # Time of deployment in seconds
t_simDur = 30 # Duration of simulation in seconds

# np.random.seed(0)  # Comment line out to provide repoqduciblility

# Create grid based on domain size
domainSize = 10
x = np.linspace(-domainSize, domainSize, (domainSize * 2) + 1)
y = np.linspace(-domainSize, domainSize, (domainSize * 2) + 1)
X, Y = np.meshgrid(x, y)

U = 0.2 * np.sin(X * 3) + 0.1 * np.random.randn((domainSize * 2) + 1, (domainSize * 2) + 1)
V = 1.0 + 0.3 * np.sin(Y * 2) + 0.1 * np.random.randn((domainSize * 2) + 1, (domainSize * 2) + 1)

# Normalize the vectors for better visualization
magnitude = np.sqrt(U**2 + V**2)
U /= magnitude.max()
V /= magnitude.max()

 # Initialize particle position
particle_x_pos = np.random.randint(-domainSize, domainSize)
particle_y_pos = np.random.randint(-domainSize, domainSize)

# Initialize drone position
drone_x_pos = np.random.randint(-domainSize, domainSize)
drone_y_pos = np.random.randint(-domainSize, domainSize)

# Update function for animation
def update(frame):
    global particle_x_pos, particle_y_pos, drone_x_pos, drone_y_pos

    plt.cla()  # Clear the axes instead of the figure to retain settings
    plt.quiver(X, Y, U, V, angles='xy', scale_units='xy')
    plt.title("Simulated Wind Field with Drone Particle Tracking")
    plt.xlabel("X - Axis")
    plt.ylabel("Y - Axis")
    plt.xticks(np.arange(-domainSize, domainSize + 1, 1))
    plt.yticks(np.arange(-domainSize, domainSize + 1, 1))
    plt.grid(True, linestyle='--', color='gray', linewidth=0.5)
    
    # Update particle position randomly
    particle_x_pos += np.random.choice([-1, 1])
    particle_y_pos += np.random.choice([-1, 1])

    if frame >= t_tod:        
        droneDist = np.sqrt((drone_x_pos - particle_x_pos)**2 + (drone_y_pos - particle_y_pos)**2) 
        plt.plot(drone_x_pos, drone_y_pos, 'bo', markersize=8, label=f'Drone')
        plt.plot(-domainSize, domainSize, 'bo', markersize=0, label = f'Drone Dist.: {droneDist:.2f} m')

        # Close into the particle's position
        if drone_x_pos < particle_x_pos:
            drone_x_pos += 1
        elif drone_x_pos > particle_x_pos:
            drone_x_pos -= 1

        if drone_y_pos < particle_y_pos:
            drone_y_pos += 1
        elif drone_y_pos > particle_y_pos:
            drone_y_pos -= 1
    
    # Keep particle & drone within bounds
    particle_x_pos = np.clip(particle_x_pos, -domainSize, domainSize)
    particle_y_pos = np.clip(particle_y_pos, -domainSize, domainSize)
    drone_x_pos = np.clip(drone_x_pos, -domainSize, domainSize)
    drone_y_pos = np.clip(drone_y_pos, -domainSize, domainSize)

    # Update particle plot
    plt.plot(particle_x_pos, particle_y_pos, 'ro', markersize=8, label=f'Particle')
    plt.plot(-domainSize, domainSize, 'ro', markersize=0, label = f'Time: {frame:.2f} s')
    
    plt.legend()  # Add legend back after clearing axes

# Create animation
fig = plt.figure(figsize=(8, 8))
ani = FuncAnimation(fig, update, frames = t_simDur, interval = 500, repeat = False) # Set interval to 1000 for real time

# Uncomment following lines to save the animation as a gif
# writer = animation.PillowWriter(fps = 2, metadata=dict(artist='Me'), bitrate=1800)
# ani.save('particleTrack.gif', writer=writer)

plt.show()
