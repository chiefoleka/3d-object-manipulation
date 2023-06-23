import os
from pytorch3d.io import IO
from iopath.common.file_io import PathManager
from torch_geometric.data import Data
from stl_converter import STLConverter

class DatasetLoader:
    def __init__(self) -> None:
        self.converter = STLConverter()

    def load(self, directory: str):
        dataset = []
        # Iterate over all files in the directory
        for filename in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, filename)):
                stl_file = os.path.join(directory, filename)
                obj_file = self.converter.convert_to_binary(stl_file)
                graph = self.converter.convert_to_graph(obj_file)
                dataset.append(graph)

        return dataset

    def convert_to_graph(self, obj_file: str):
        path_manager = PathManager()
        path_manager.set_logging(False)
        generated_mesh = IO(path_manager=path_manager).load_mesh(obj_file, include_textures=False)

        # it is the position of the vertices
        edges = generated_mesh.edges_packed().t().contiguous()
        verts = generated_mesh.verts_packed()

        # the vertice normal that we can add later
        # verts_normal = generated_mesh.verts_normals_packed()
        # faces = generated_mesh.faces_packed()

        # x will be the position of each vertex (node)
        graph = Data(x=verts, edge_index=edges)

        return graph
