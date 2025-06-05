import vtk
import numpy as np

def parse_vtk_file(vtk_file_path):
    """
    解析 VTK 文件，提取顶点和三角形数据。

    参数:
    - vtk_file_path (str): VTK 文件的路径。

    返回:
    - vertices (numpy.ndarray): 顶点坐标数组，形状为 (num_vertices, 3)。
    - triangles (numpy.ndarray): 三角形索引数组，形状为 (num_triangles, 3)。
    """
    # 读取 VTK 文件
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_file_path)
    reader.Update()

    # 获取网格数据
    mesh = reader.GetOutput()

    # 获取顶点数据
    points = mesh.GetPoints()
    # 获取面（多边形）数据，主要是三角形
    polys = mesh.GetPolys()

    # 提取顶点
    vertices = np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())])

    # 提取三角形
    triangles = []
    polys.InitTraversal()
    id_list = vtk.vtkIdList()
    while polys.GetNextCell(id_list):
        # 获取三角形的三个顶点的索引
        if id_list.GetNumberOfIds() == 3:  # 确保是三角形
            triangles.append([id_list.GetId(i) for i in range(3)])

    triangles = np.array(triangles)

    return vertices, triangles


def write_gocad_tsurf(vertices, triangles, output_file_path):
    """
    将顶点和三角形数据写入 GOCAD TSurf 文件。

    参数:
    - vertices (numpy.ndarray): 顶点坐标数组。
    - triangles (numpy.ndarray): 三角形索引数组。
    - output_file_path (str): 输出的 GOCAD TSurf 文件路径。
    """
    with open(output_file_path, 'w') as file:
        # 写入 GOCAD 文件头
        file.write("GOCAD TSurf 1\n")
        file.write("HEADER {\n")
        file.write("name: GOCAD_Tsurf\n")
        file.write("}\n")

        # 写入顶点数据
        file.write("VRTX {0}\n".format(len(vertices)))
        for i, vertex in enumerate(vertices):
            file.write("VRTX {0} {1} {2} {3}\n".format(i + 1, vertex[0], vertex[1], vertex[2]))

        # 写入三角形数据
        file.write("TRGL {0}\n".format(len(triangles)))
        for i, triangle in enumerate(triangles):
            file.write("TRGL {0} {1} {2}\n".format(triangle[0] + 1, triangle[1] + 1, triangle[2] + 1))

        # 写入文件尾
        file.write("END\n")


def vtk_to_gocad(vtk_file_path, output_file_path):
    """
    将 VTK 文件转换为 GOCAD TSurf 文件。

    参数:
    - vtk_file_path (str): 输入的 VTK 文件路径。
    - output_file_path (str): 输出的 GOCAD TSurf 文件路径。
    """
    vertices, triangles = parse_vtk_file(vtk_file_path)
    write_gocad_tsurf(vertices, triangles, output_file_path)


# 示例用法
if __name__ == "__main__":
    # 假设您有一个名为 'example.vtk' 的 VTK 文件
    vtk_file = 'layer_2.vtk'  # 替换为您的 .vtk 文件路径
    output_gocad_file = 'layer_2.ts'  # 输出 GOCAD TSurf 文件路径

    # 调用函数进行转换
    vtk_to_gocad(vtk_file, output_gocad_file)
