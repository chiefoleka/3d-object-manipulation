import torch
import torch_geometric
import trimesh
import numpy as np

def stl_to_3d_graph(stl_file):
    mesh = trimesh.load_mesh(stl_file)

    vertices = torch.tensor(mesh.vertices, dtype=torch.float)
    faces = torch.tensor(mesh.faces, dtype=torch.long).t().contiguous()

    data = torch_geometric.data.Data(pos=vertices, face=faces)

    return data

def stl_to_pointcloud(stl_file):
    mesh = trimesh.load_mesh(stl_file)

    # Extract vertex positions
    vertices = np.asarray(mesh.vertices)

    return vertices
