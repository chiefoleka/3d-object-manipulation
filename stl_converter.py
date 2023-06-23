import os
import sys
import stl
from pytorch3d.io import IO
from iopath.common.file_io import PathManager
from torch_geometric.data import Data

class STLConverter:
    def _print_error(self, *str):
        print('ERROR: '),
        for i in str:
            print(i),
        print()
        sys.exit()

    def _is_file_stl(self, stl_file_name):
        if stl_file_name.split('.')[-1].lower() != 'stl':
            self._print_error(stl_file_name, ': The file is not an .stl file.')

        if not os.path.exists(stl_file_name):
            self._print_error(stl_file_name, ": The file doesn't exist.")

    def _ensure_output_file(self, input_file_name, output_filename=None, output_subdir='obj', format='obj'):
        if output_filename is None:
            input_paths = input_file_name.split('/')
            output_paths = input_paths[:-1]
            output_paths.append(output_subdir)

            # create the new file in obj path
            filename = input_paths[-1].split(".")[0]
            output_paths.append(f'{filename}.{format}')
            output_filename = '/'.join(output_paths)

        return output_filename

    def _get_point_id(self, point, list):
        for i, pts in enumerate(list):
            if pts[0] == point[0] and pts[1] == point[1] and pts[2] == point[2]:
                #obj start to count at 1
                return i + 1
            
        list.append(point)
        #obj start to count at 1
        return len(list)

    def _convert_ascii_to_obj(self, stl_file_name, obj_filename):
        pointList = []
        facetList = []

        # start reading the STL file
        stl_file = open(stl_file_name, 'r')
        line = stl_file.readline()
        line_number = 1

        while line != '':
            tab = line.strip().split()        
            if len(tab) > 0:
                if tab[0] == 'facet':
                    vertices = []
                    normal = list(map(float, tab[2:]))

                    while tab[0] != 'endfacet':
                        if tab[0] == 'vertex':
                            pts = list(map(float, tab[1:]))
                            vertices.append(self._get_point_id(pts, pointList))

                        line = stl_file.readline()
                        line_number = line_number + 1
                        tab = line.strip().split()

                    if len(vertices) == 0:
                        self._print_error('Unvalid facet description at line ', line_number)

                    facetList.append({'vertices': vertices, 'normal': normal})

            line = stl_file.readline()
            line_number = line_number + 1
                
        stl_file.close()

        # Write the target file
        obj_file = open(obj_filename, 'w')
        obj_file.write('# File type: ASCII OBJ\n')
        obj_file.write('# Generated from ' + os.path.basename(stl_file_name) + '\n')

        for pts in pointList:
            obj_file.write('v ' + ' '.join(list(map(str, pts))) + '\n')

        for f in facetList:
            obj_file.write('f ' + ' '.join(list(map(str, f['vertices']))) + '\n')

        obj_file.close()

        return obj_filename

    # convert ascii stl file to obj file
    def _convert_binary_to_ascii(self, binary_filename, ascii_filename=None):
        ascii_filename = self._ensure_output_file(binary_filename, ascii_filename, format='ascii.stl')

        stl_mesh = stl.mesh.Mesh.from_file(binary_filename, mode=stl.Mode.BINARY)
        stl_mesh.save(ascii_filename, mode = stl.Mode.ASCII)

        return ascii_filename

    def convert_to_obj(self, input_filename, obj_filename=None):
        self._is_file_stl(input_filename)
        obj_filename = self._ensure_output_file(input_filename, obj_filename)

        self._convert_ascii_to_obj(input_filename, obj_filename)

    def convert_to_binary(self, binary_filename, obj_filename=None):
        self._is_file_stl(binary_filename)

        obj_filename = self._ensure_output_file(binary_filename)

        if not os.path.exists(obj_filename):
            ascii_filename = self._convert_binary_to_ascii(binary_filename)

            obj_filename = self._convert_ascii_to_obj(ascii_filename, obj_filename)
            os.remove(ascii_filename)

        return obj_filename
    
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
