import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

nc = netCDF4.Dataset('FireTemp_field.nc', 'r')

temperature = nc.variables['temperature'][:]



def update(frame):
    plt.clf()  # Clear the current figure
    plt.imshow(temperature[frame], cmap='hot', interpolation='nearest')
    plt.title(f"Temperature Field at Frame {frame}")
    plt.colorbar(label='Temperature (K)')
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")

# Create the animation
fig = plt.figure(figsize=(8, 8))
ani = FuncAnimation(fig, update, frames=temperature.shape[0], interval=100)
plt.show()
