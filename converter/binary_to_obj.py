import os
import stl
from utils.logger import Logger
from converter.helper import Helper


class BinaryToObj:
    def _get_point_id(self, point, list):
        for i, pts in enumerate(list):
            if pts[0] == point[0] and pts[1] == point[1] and pts[2] == point[2]:
                # obj start to count at 1
                return i + 1

        list.append(point)
        # obj start to count at 1
        return len(list)

    def _convert_ascii_to_obj(self, stl_file_name, obj_filename):
        pointList = []
        facetList = []

        # start reading the STL file
        stl_file = open(stl_file_name, "r")
        line = stl_file.readline()
        line_number = 1

        while line != "":
            tab = line.strip().split()
            if len(tab) > 0:
                if tab[0] == "facet":
                    vertices = []
                    normal = list(map(float, tab[2:]))

                    while tab[0] != "endfacet":
                        if tab[0] == "vertex":
                            pts = list(map(float, tab[1:]))
                            vertices.append(self._get_point_id(pts, pointList))

                        line = stl_file.readline()
                        line_number = line_number + 1
                        tab = line.strip().split()

                    if len(vertices) == 0:
                        Logger.error("Unvalid facet description at line ", line_number)

                    facetList.append({"vertices": vertices, "normal": normal})

            line = stl_file.readline()
            line_number = line_number + 1

        stl_file.close()

        # Write the target file
        obj_file = open(obj_filename, "w")
        obj_file.write("# File type: ASCII OBJ\n")
        obj_file.write("# Generated from " + os.path.basename(stl_file_name) + "\n")

        for pts in pointList:
            obj_file.write("v " + " ".join(list(map(str, pts))) + "\n")

        for f in facetList:
            obj_file.write("f " + " ".join(list(map(str, f["vertices"]))) + "\n")

        obj_file.close()

        return obj_filename

    # convert ascii stl file to obj file
    def _convert_binary_to_ascii(self, stl_filename, ascii_filename=None):
        ascii_filename = Helper.ensure_file_exists(
            stl_filename, ascii_filename, format="ascii.stl"
        )

        stl_mesh = stl.mesh.Mesh.from_file(stl_filename, mode=stl.Mode.AUTOMATIC)
        stl_mesh.save(ascii_filename, mode=stl.Mode.ASCII)

        return ascii_filename

    def convert_to_obj(self, stl_filename):
        Helper.check_file_is_stl(stl_filename)
        obj_filename = Helper.ensure_file_exists(stl_filename)

        if not os.path.exists(obj_filename):
            ascii_filename = self._convert_binary_to_ascii(stl_filename)

            obj_filename = self._convert_ascii_to_obj(ascii_filename, obj_filename)
            os.remove(ascii_filename)

        return obj_filename
