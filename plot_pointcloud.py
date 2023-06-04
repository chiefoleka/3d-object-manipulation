import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

def stl_to_pointcloud(stl_file):
    mesh = trimesh.load_mesh(stl_file)
    vertices = np.asarray(mesh.vertices)
    return vertices

stl_file = 'data/sample.STL'
pointcloud = stl_to_pointcloud(stl_file)

# Visualize point cloud using Matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

# Visualize point cloud using Plotly
fig = go.Figure(data=[go.Scatter3d(x=pointcloud[:, 0], y=pointcloud[:, 1], z=pointcloud[:, 2], mode='markers')])
fig.show()
