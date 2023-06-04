import torch
from pytorch3d.io import IO
from iopath.common.file_io import PathManager
from stl_to_obj_converter import convert_binary
from pytorch3d.structures import Pointclouds
import torch_geometric

stl_file = 'data/sample.STL'
obj_file = 'data/sample.obj'

convert_binary(stl_file, obj_file)

path_manager = PathManager()
path_manager.set_logging(False)
generated_mesh = IO(path_manager=path_manager).load_mesh(obj_file, include_textures=False)

vertices = generated_mesh.verts_packed()
faces = generated_mesh.faces_packed()

# Convert the tensors to PyTorch data types if necessary 
vertices = vertices.to(torch.float32) 
faces = faces.to(torch.int64)

graph = torch_geometric.data.Data(pos=vertices, face=faces)

vertices_array = vertices
if vertices_array.dim() == 2 and vertices_array.size(1) == 3:
    vertices_array = vertices_array.unsqueeze(0)

point_clouds = Pointclouds(points=vertices_array)
