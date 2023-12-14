import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

r = 5.0
a, b, c = (0.0, 0.0, 0.0)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-10,10)
ax.set_ylim(-10,10)
ax.set_zlim(-10,10)

phirange = np.linspace(0, 2 * np.pi, 8) #to make a full circle

x = a + r * np.cos(phirange)
y = b + r * np.sin(phirange)
z = c

center_x = np.array([0])
center_y = np.array([0])
center_z = np.array([0])

ax.scatter(x, y, z)
ax.scatter(center_x, center_y, center_z, c="r")
plt.show()
