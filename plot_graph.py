import trimesh
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def stl_to_graph(stl_file):
    mesh = trimesh.load_mesh(stl_file)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    # Create a graph
    graph = nx.Graph()

    # Add nodes
    for i, vertex in enumerate(vertices):
        graph.add_node(i, pos=(vertex[0], vertex[1]))

    # Add edges
    for face in faces:
        graph.add_edge(face[0], face[1])
        graph.add_edge(face[1], face[2])
        graph.add_edge(face[2], face[0])

    return graph

stl_file = 'data/sample.STL'
graph = stl_to_graph(stl_file)

pos = nx.get_node_attributes(graph, 'pos')

# Convert node positions to 2D coordinates
x = [pos[node][0] for node in graph.nodes()]
y = [pos[node][1] for node in graph.nodes()]

# Plot the graph
plt.figure(figsize=(8, 8))
plt.scatter(x, y, s=10)
plt.gca().set_aspect('equal')
plt.show()
