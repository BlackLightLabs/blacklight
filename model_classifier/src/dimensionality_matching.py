import numpy as np
import matplotlib.pyplot as plt


twod_array = np.array([[1, 2, 5**0.5], [4, 5, (16 + 25)**0.5], [7, 8, (49 + 64)**0.5]])

threed_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(threed_array[:,0], threed_array[:,1], threed_array[:,2], c='r', marker='o')
ax.scatter(twod_array[:,0], twod_array[:,1], twod_array[:,2], c='b', marker='o')
plt.show()

