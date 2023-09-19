import vtk
from pathlib import Path

def reducestl(filename, ratio=1.0, output_filename=None):
    output_filename = output_filename if output_filename else f"{Path(filename).stem}_reduced.stl"

    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename)
    reader.Update()

    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputData(reader.GetOutput())
    triangles.Update()

    decimate = vtk.vtkQuadricDecimation()
    decimate.SetInputData(triangles.GetOutput())
    decimate.SetTargetReduction(ratio)
    # decimate.PreserveTopologyOn()
    decimate.Update()

    writer = vtk.vtkSTLWriter()
    writer.SetFileName(output_filename)
    writer.SetInputData(decimate.GetOutput())
    writer.Write()


if __name__ == '__main__':
    ratio = 1
    i = 4
    loop = 50
    filename = f"/home/puchaud/Projets_Python/bioptim_exo/models/geometry/Piece{i}.STL"
    output_filename = f"/home/puchaud/Projets_Python/bioptim_exo/models/geometry/{Path(filename).stem}_reduced.stl"
    for _ in range(loop):
        filename = filename if i == 0 else output_filename
        reducestl(
            filename,
            ratio=ratio,
            output_filename=f"/home/puchaud/Projets_Python/bioptim_exo/models/geometry/{Path(filename).stem}_reduced.stl"
        )
    filename = f"/home/puchaud/Projets_Python/bioptim_exo/models/geometry/Piece{i}.STL"
    output_filename = f"/home/puchaud/Projets_Python/bioptim_exo/models/geometry/{Path(filename).stem}_reduced.stl"
    for _ in range(loop):
        for i in range(5, 8):
            filename = filename if i == 0 else output_filename
            reducestl(
                filename,
                ratio=ratio,
                output_filename=f"/home/puchaud/Projets_Python/bioptim_exo/models/geometry/{Path(filename).stem}_reduced.stl"
            )
