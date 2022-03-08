import vtk
import glob, os


def STL2vtp(name: str):
    """
        Convert a stl file into vtp file

        Parameters
        ----------
        name: str
            The name of the file
    """
    readerSTL = vtk.vtkSTLReader()
    readerSTL.SetFileName(name + ".STL")
    # 'update' the reader i.e. read the .stl file
    readerSTL.Update()
    polydata = readerSTL.GetOutput()

    # if I need to scale the mesh
    # polydata = readerSTL.GetOutputPort()
    #
    # transform = vtk.vtkTransform()
    # transform.Scale((0.001, 0.001, 0.001))
    #
    # transformFilter = vtk.vtkTransformPolyDataFilter()
    # transformFilter.SetInputConnection(polydata)
    # transformFilter.SetTransform(transform)
    # transformFilter.Update()
    #
    # polydata = transformFilter.GetOutput()

    writter = vtk.vtkXMLPolyDataWriter()
    writter.SetFileName(name + ".vtp")
    writter.SetInputData(polydata)
    writter.SetDataModeToAscii()

    # writter.SetFileTypeToASCII()
    res = writter.Write()

    return res


if __name__ == "__main__":

    mypath = "/home/puchaud/Projets_Python/My_bioptim_examples/Kinova_arm/geom"
    os.chdir(mypath)
    fileList = []
    for file in glob.glob("*.STL"):
        print(os.path.splitext(file)[0])
        fileList.append(os.path.splitext(file)[0])

    for ii in fileList:
        if "m.STL" in ii:
            STL2vtp(ii)


