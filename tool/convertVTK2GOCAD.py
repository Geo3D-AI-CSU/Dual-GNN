import vtk
import numpy as np

def parse_vtk_file(vtk_file_path):

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_file_path)
    reader.Update()

    mesh = reader.GetOutput()

    points = mesh.GetPoints()
    polys = mesh.GetPolys()

    vertices = np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())])

    triangles = []
    polys.InitTraversal()
    id_list = vtk.vtkIdList()
    while polys.GetNextCell(id_list):
        if id_list.GetNumberOfIds() == 3:  
            triangles.append([id_list.GetId(i) for i in range(3)])

    triangles = np.array(triangles)

    return vertices, triangles


def write_gocad_tsurf(vertices, triangles, output_file_path):

    with open(output_file_path, 'w') as file:
        file.write("GOCAD TSurf 1\n")
        file.write("HEADER {\n")
        file.write("name: GOCAD_Tsurf\n")
        file.write("}\n")
        file.write("VRTX {0}\n".format(len(vertices)))
        for i, vertex in enumerate(vertices):
            file.write("VRTX {0} {1} {2} {3}\n".format(i + 1, vertex[0], vertex[1], vertex[2]))

        file.write("TRGL {0}\n".format(len(triangles)))
        for i, triangle in enumerate(triangles):
            file.write("TRGL {0} {1} {2}\n".format(triangle[0] + 1, triangle[1] + 1, triangle[2] + 1))

        file.write("END\n")


def vtk_to_gocad(vtk_file_path, output_file_path):

    vertices, triangles = parse_vtk_file(vtk_file_path)
    write_gocad_tsurf(vertices, triangles, output_file_path)


if __name__ == "__main__":
    vtk_file = 'layer_2.vtk'  
    output_gocad_file = 'layer_2.ts'  
    vtk_to_gocad(vtk_file, output_gocad_file)

