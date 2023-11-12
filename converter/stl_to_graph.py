import re
import torch
import numpy
import math
from pytorch3d.io import IO
from iopath.common.file_io import PathManager
from torch_geometric.data import Data


class StlToGraph:
    def convert_to_graph(self, obj_file: str, params: dict):
        path_manager = PathManager()
        path_manager.set_logging(False)
        generated_mesh = IO(path_manager=path_manager).load_mesh(
            obj_file, include_textures=False
        )

        # it is the position of the vertices
        edges = generated_mesh.edges_packed().t().contiguous()
        verts = generated_mesh.verts_packed()

        # from the params in each directory, get the number in the file
        param_id = re.findall(r"\d+", obj_file)[0]
        # try this without converting to a long so extra info is not dropped
        y = torch.tensor(params[param_id]).float()

        # the vertice normal that we can add later
        faces = generated_mesh.faces_packed()

        # x will be the position of each vertex (node)
        filename = obj_file.split("/")[-1].split(".")[0]

        """
        first, for each vertices, find corresponding faces (all the faces that have this
        vertice in it), get the mean value of those faces, corresponding to each vertice
        (mean value of the norms for each face)

        x, y, z, x_face_norm, y_face_norm, z_face_norm
        """

        face_mapping = {}
        for face in faces:
            a, b, c = face.numpy()

            # https://www.khronos.org/opengl/wiki/Calculating_a_Surface_Normal
            norm = numpy.cross(verts[b] - verts[a], verts[c] - verts[b])

            # For all faces, map the nodes to faces, so we have a single array that
            # represents all the faces that a node exists in
            if a not in face_mapping:
                face_mapping[a] = []
            if b not in face_mapping:
                face_mapping[b] = []
            if c not in face_mapping:
                face_mapping[c] = []

            face_mapping[a].append(norm)
            face_mapping[b].append(norm)
            face_mapping[c].append(norm)

        new_verts = [None] * len(verts)
        for key, face_norms in face_mapping.items():
            sum_x, sum_y, sum_z = 0, 0, 0

            # take the mean of all the norms for each param
            for fn in face_norms:
                sum_x, sum_y, sum_z = sum_x + fn[0], sum_y + fn[1], sum_z + fn[2]

            # mean for the norms broken into x, y, z
            x_mean = sum_x / len(face_norms)
            y_mean = sum_y / len(face_norms)
            z_mean = sum_z / len(face_norms)

            # length of vector = math.sqrt(x^2 + y^2 + z^2)
            length_of_vector = math.sqrt(x_mean ** 2 + y_mean ** 2 + z_mean ** 2)

            x_norm = x_mean / length_of_vector
            y_norm = y_mean / length_of_vector
            z_norm = z_mean / length_of_vector

            # convert tensor to list to make adding the x, y, z norms easy
            verts_list = verts[key].tolist()

            # x, y, z, x_face_norm, y_face_norm, z_face_norm
            verts_list.append(torch.tensor(x_norm))
            verts_list.append(torch.tensor(y_norm))
            verts_list.append(torch.tensor(z_norm))

            # verts = [*, 3]
            # new_verts = [*, 6]
            new_verts[key] = verts_list

        return Data(
            x=verts,
            x_norms=torch.tensor(new_verts).float(),
            edge_index=edges,
            y=y,
            filename=filename,
        )
