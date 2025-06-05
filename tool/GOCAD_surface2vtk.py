import numpy as np
import pyvista as pv


def parse_gocad_tsurf(tsurf_file_path):
    """
    解析 GOCAD TSurf .ts 文件，提取顶点和三角形数据。

    参数:
    - tsurf_file_path (str): .ts 文件的路径。

    返回:
    - vertices (numpy.ndarray): 顶点坐标数组，形状为 (num_vertices, 3)。
    - triangles (numpy.ndarray): 三角形索引数组，形状为 (num_triangles, 3)。
    """
    vertices = []
    triangles = []

    # 修改这里，指定编码为 'latin-1' 或 'utf-8'
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
    """
    使用 PyVista 创建三角网格。

    参数:
    - vertices (numpy.ndarray): 顶点坐标数组。
    - triangles (numpy.ndarray): 三角形索引数组。

    返回:
    - mesh (pv.PolyData): PyVista 三角网格对象。
    """
    # PyVista 需要三角形数据以 [3, idx1, idx2, idx3] 的形式
    faces = np.hstack([np.full((triangles.shape[0], 1), 3), triangles]).flatten()
    mesh = pv.PolyData(vertices, faces)
    return mesh


def visualize_mesh(mesh, color='lightblue', show_edges=False, show_normals=False):
    """
    可视化 PyVista 网格。

    参数:
    - mesh (pv.PolyData): 要可视化的 PyVista 网格对象。
    - color (str or tuple): 网格颜色。
    - show_edges (bool): 是否显示边缘。
    - show_normals (bool): 是否显示法线。
    """
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color=color, show_edges=show_edges,edge_color="black")

    if show_normals:
        mesh_copy = mesh.copy()
        mesh_copy.compute_normals(inplace=True)
        plotter.add_arrows(mesh_copy['Normals'], mag=0.1, color='red')

    plotter.show()


def tsurf_to_pyvista(tsurf_file_path, visualize=True, show_normals=False):
    """
    将 GOCAD TSurf .ts 文件转换为 PyVista.PolyData 并进行可视化。

    参数:
    - tsurf_file_path (str): .ts 文件的路径。
    - visualize (bool): 是否进行可视化。
    - show_normals (bool): 是否显示法线。

    返回:
    - mesh (pv.PolyData): 生成的 PyVista 三角网格对象。
    """
    vertices, triangles = parse_gocad_tsurf(tsurf_file_path)
    mesh = create_pyvista_mesh(vertices, triangles)

    if visualize:
        visualize_mesh(mesh, show_normals=show_normals)

    return mesh
# 示例用法
if __name__ == "__main__":
    # 假设您有一个名为 'example.ts' 的 TSurf 文件
    tsurf_file = 'E:\XZHThesisExperiment\GNN_experiment\GCN_model/tetra_output_files\hrbf\help/f.ts'

    # 调用函数进行转换和可视化
    mesh = tsurf_to_pyvista(tsurf_file, visualize=True)
    mesh.save('E:\XZHThesisExperiment\GNN_experiment\GCN_model/tetra_output_files\hrbf\help/f.vtk')
