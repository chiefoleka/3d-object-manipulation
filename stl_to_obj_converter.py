import os
import sys
import os.path
import stl

def print_error(*str):
    print('ERROR: '),
    for i in str:
        print(i),
    print()
    sys.exit()

def is_file_stl(stl_file_name):
    if stl_file_name.split('.')[-1].lower() != 'stl':
        print_error(stl_file_name, ': The file is not an .stl file.')

    if not os.path.exists(stl_file_name):
        print_error(stl_file_name, ": The file doesn't exist.")

def ensure_output_file(input_file_name, output_filename=None, format='obj'):
    if output_filename is None:
        filename = input_file_name.split(".")[0]
        output_filename = f'{filename}.{format}'

    return output_filename

def get_point_id(point, list):
    for i, pts in enumerate(list):
        if pts[0] == point[0] and pts[1] == point[1] and pts[2] == point[2]:
            #obj start to count at 1
            return i + 1
        
    list.append(point)
    #obj start to count at 1
    return len(list)

def _convert_ascii_to_obj(stl_file_name, obj_filename):
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
                        vertices.append(get_point_id(pts, pointList))

                    line = stl_file.readline()
                    line_number = line_number + 1
                    tab = line.strip().split()

                if len(vertices) == 0:
                    print_error('Unvalid facet description at line ', line_number)

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
def _convert_binary_to_ascii(binary_filename, ascii_filename=None):
    ascii_filename = ensure_output_file(binary_filename, ascii_filename, 'ascii.stl')

    stl_mesh = stl.mesh.Mesh.from_file(binary_filename, mode=stl.Mode.BINARY)
    stl_mesh.save(ascii_filename, mode = stl.Mode.ASCII)

    return ascii_filename

def convert(input_filename, obj_filename=None):
    is_file_stl(input_filename)
    obj_filename = ensure_output_file(input_filename, obj_filename)

    _convert_ascii_to_obj(input_filename, obj_filename)

def convert_binary(binary_filename, obj_filename=None):
    is_file_stl(binary_filename)

    obj_filename = ensure_output_file(obj_filename)
    ascii_filename = _convert_binary_to_ascii(binary_filename)

    obj_filename = _convert_ascii_to_obj(ascii_filename, obj_filename)
    os.remove(ascii_filename)

    return obj_filename
