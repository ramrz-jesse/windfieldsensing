import matplotlib.pyplot as plt
import numpy as np

ax = plt.figure().add_subplot(projection='3d')

# Create customizable domain

domainSize = 10

# Adjust the meshgrid to plot a vector every two ticks
x = np.arange(0, domainSize + 1, 2)
y = np.arange(0, domainSize + 1, 2)
z = np.arange(0, domainSize + 1, 2)
X, Y, Z = np.meshgrid(x, y, z)

u = 1.0 + 0.3 * np.sin(Y * 2) + 0.1 * np.random.randn(*Y.shape)
v = 0.1 * np.sin(X * 2) + 0.1 * np.random.randn(*X.shape)
w = 0.1 * np.sin(Z * 2) + 0.1 * np.random.randn(*Z.shape)

ax.quiver(X, Y, Z, u, v, w, normalize=True)
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('Wind Field')
ax.set_xlim([0, domainSize + 1])
ax.set_ylim([0, domainSize + 1])
ax.set_zlim([0, domainSize + 1])

# Set the aspect ratio to be equal
ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1

# Plot a single point at the origin
ax.scatter(0, 0, 0, color='red', s=100, label='Origin')

# Add a legend
ax.legend()

plt.show()