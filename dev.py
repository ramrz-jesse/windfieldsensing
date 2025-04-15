import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

np.random.seed(0)  # for reproducibility

# Create a 32x32 grid
x = np.linspace(0, 2 * np.pi, 32)
y = np.linspace(0, 2 * np.pi, 32)
X, Y = np.meshgrid(x, y)

U = 1.0 + 0.3 * np.sin(Y * 2) + 0.1 * np.random.randn(32, 32)
# U = np.sin(Y) * np.random.randn(32, 32)
V = 0.2 * np.sin(X * 3) + 0.1 * np.random.randn(32, 32)
# V = np.sin(X) * np.random.randn(32, 32)
# Normalize the vectors for better visualization
magnitude = np.sqrt(U**2 + V**2)
U /= magnitude.max()
V /= magnitude.max()


# Function to update the point's position
def update(frame):
    start_y = 0
    plt.clf()  # Clear the current figure
    plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=5)
    plt.title("Simulated Wind Field with Sweeping Point")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    if frame // 32 % 2 == 0:  # Left to right
        new_x = (frame % 32) * (2 * np.pi / 31)
    else:  # Right to left
        new_x = (31 - frame % 32) * (2 * np.pi / 31)

    # Move down when reaching the edges
    new_y = start_y + (frame // 32) * (2 * np.pi / 31)
    new_y = new_y % (2 * np.pi)  # Wrap around if it exceeds the grid
    
    grid_x = int(np.clip(np.round((new_x / (2 * np.pi)) * 31), 0, 31))  # Map to grid index
    grid_y = int(np.clip(np.round((new_y / (2 * np.pi)) * 31), 0, 31))  # Map to grid index
    # convert U and V to NSEW and get speed
    U_speed = U[grid_y, grid_x] * 10  # Scale for visualization
    V_speed = V[grid_y, grid_x] * 10  # Scale for visualization
    
    # Add noise to speed and direction
    noise_speed = np.random.normal(0, 0.2)  # Mean 0, standard deviation 0.2
    noise_direction = np.random.normal(0, 1)  # Mean 0, standard deviation 1 degrees

    speed = np.sqrt(U_speed**2 + V_speed**2)    

    direction = np.arctan2(V_speed, U_speed)  # Angle in radians
    direction_deg = np.degrees(direction)  # Convert to degrees

    speed += noise_speed
    direction_deg += noise_direction

    print(f"Frame: {frame},Position: {grid_x,grid_y} Speed: {speed:.2f}, Direction: {direction_deg:.2f} degrees")
    
    drone_count = 8
    for i in range(drone_count):
        drone_x = new_x
        drone_y = (new_y + i * (2 * np.pi / drone_count)) % (2 * np.pi)
        plt.plot(drone_x, drone_y, 'bo', markersize=5)  # Plot the drones
        # Print data for each drone
        drone_grid_x = int(np.clip(np.round((drone_x / (2 * np.pi)) * 31), 0, 31))
        drone_grid_y = int(np.clip(np.round((drone_y / (2 * np.pi)) * 31), 0, 31))
        drone_U_speed = U[drone_grid_y, drone_grid_x] * 10  # Scale for visualization
        drone_V_speed = V[drone_grid_y, drone_grid_x] * 10  # Scale for visualization

        # Add noise to speed and direction
        drone_noise_speed = np.random.normal(0, 0.2)  # Mean 0, standard deviation 0.2
        drone_noise_direction = np.random.normal(0, 1)  # Mean 0, standard deviation 1 degrees

        drone_speed = np.sqrt(drone_U_speed**2 + drone_V_speed**2)
        drone_direction = np.arctan2(drone_V_speed, drone_U_speed)  # Angle in radians
        drone_direction_deg = np.degrees(drone_direction)  # Convert to degrees

        drone_speed += drone_noise_speed
        drone_direction_deg += drone_noise_direction

        print(f"Drone {i+1} | Position: {drone_grid_x, drone_grid_y} | Speed: {drone_speed:.2f} | Direction: {drone_direction_deg:.2f} degrees")
    plt.plot(new_x, new_y, 'ro', markersize=5)  # Plot the sweeping point

# Create the animation
fig = plt.figure(figsize=(8, 8))
ani = FuncAnimation(fig, update, frames=1000, interval=100)
plt.show()