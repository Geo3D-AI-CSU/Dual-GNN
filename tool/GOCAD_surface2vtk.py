import numpy as np
import pyvista as pv


def parse_gocad_tsurf(tsurf_file_path):
    vertices = []
    triangles = []
    with open(tsurf_file_path, 'r', encoding='latin-1') as file:
        for line in file:
            line = line.strip()
            if line.startswith("VRTX"):
                parts = line.split()
                if len(parts) >= 5:
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    vertices.append([x, y, z])
            elif line.startswith("TRGL"):
                parts = line.split()
                if len(parts) >= 4:
                    idx1, idx2, idx3 = int(parts[1]) - 1, int(parts[2]) - 1, int(parts[3]) - 1
                    triangles.append([idx1, idx2, idx3])

    vertices = np.array(vertices)
    triangles = np.array(triangles)

    return vertices, triangles


def create_pyvista_mesh(vertices, triangles):
    faces = np.hstack([np.full((triangles.shape[0], 1), 3), triangles]).flatten()
    mesh = pv.PolyData(vertices, faces)
    return mesh


def visualize_mesh(mesh, color='lightblue', show_edges=False, show_normals=False):
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color=color, show_edges=show_edges,edge_color="black")

    if show_normals:
        mesh_copy = mesh.copy()
        mesh_copy.compute_normals(inplace=True)
        plotter.add_arrows(mesh_copy['Normals'], mag=0.1, color='red')

    plotter.show()


def tsurf_to_pyvista(tsurf_file_path, visualize=True, show_normals=False):
    vertices, triangles = parse_gocad_tsurf(tsurf_file_path)
    mesh = create_pyvista_mesh(vertices, triangles)

    if visualize:
        visualize_mesh(mesh, show_normals=show_normals)

    return mesh
if __name__ == "__main__":
    tsurf_file = 'E:\XZHThesisExperiment\GNN_experiment\GCN_model/tetra_output_files\hrbf\help/f.ts'
    mesh = tsurf_to_pyvista(tsurf_file, visualize=True)
    mesh.save('E:\XZHThesisExperiment\GNN_experiment\GCN_model/tetra_output_files\hrbf\help/f.vtk')

