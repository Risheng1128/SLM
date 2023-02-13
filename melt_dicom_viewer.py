import sys
import vtk

from PyQt5 import QtWidgets
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

"""
A minimal example of a DICOM viewer using VTK and Qt.
Rasmus R. Paulsen. DTU Compute. 2022
"""
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, dicom_folder, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.frame = QtWidgets.QFrame()
        self.vl = QtWidgets.QVBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.vl.addWidget(self.vtkWidget)
        self.image_viewer = None
        self.slice_text_mapper = None
        self.ren_win = None
        self.folder = dicom_folder
        self.setup_screen_things()
        self.frame.setLayout(self.vl)
        self.setCentralWidget(self.frame)
        self.show()
        self.iren.Initialize()

    def setup_screen_things(self):
        print(f"Reading DICOM files from {self.folder}")
        reader = vtk.vtkDICOMImageReader()
        reader.SetDirectoryName(self.folder)
        reader.Update()
        print("Reading done")

        self.image_viewer = vtk.vtkImageViewer2()
        self.image_viewer.SetInputConnection(reader.GetOutputPort())
        self.ren_win = self.vtkWidget.GetRenderWindow()
        self.ren_win.AddRenderer(self.image_viewer.GetRenderer())
        self.image_viewer.SetRenderWindow(self.ren_win)
        self.iren = self.image_viewer.GetRenderWindow().GetInteractor()
        self.style = vtk.vtkInteractorStyleImage()
        self.iren.SetInteractorStyle(self.style)
        self.iren.AddObserver('KeyPressEvent', self.keypress_callback, 1.0)

        slice_text_prop = vtk.vtkTextProperty()
        slice_text_prop.SetFontFamilyToCourier()
        slice_text_prop.SetFontSize(20)
        slice_text_prop.SetVerticalJustificationToBottom()
        slice_text_prop.SetJustificationToLeft()

        self.slice_text_mapper = vtk.vtkTextMapper()
        msg = f"{self.image_viewer.GetSlice()} / \
                {self.image_viewer.GetSliceMax()}"
        self.slice_text_mapper.SetInput(msg)
        self.slice_text_mapper.SetInput(msg)
        self.slice_text_mapper.SetTextProperty(slice_text_prop)

        slice_text_actor = vtk.vtkActor2D()
        slice_text_actor.SetMapper(self.slice_text_mapper)
        slice_text_actor.SetPosition(15, 10)

        self.image_viewer.GetRenderer().AddActor2D(slice_text_actor)
        self.image_viewer.GetRenderer().ResetCamera()
        self.image_viewer.Render()

    def keypress_callback(self, obj, ev):
        key = obj.GetKeySym()
        if key == 'Up':
            cur_slice = self.image_viewer.GetSlice()
            if cur_slice < self.image_viewer.GetSliceMax():
                self.image_viewer.SetSlice(cur_slice + 1)
        if key == 'Down':
            cur_slice = self.image_viewer.GetSlice()
            if cur_slice > self.image_viewer.GetSliceMin():
                self.image_viewer.SetSlice(cur_slice - 1)

        msg = f"{self.image_viewer.GetSlice()} / \
                {self.image_viewer.GetSliceMax()}"
        self.slice_text_mapper.SetInput(msg)
        self.image_viewer.Render()

def run_qt_window():
    app = QtWidgets.QApplication(sys.argv)
    folder = None
    filedialog = QtWidgets.QFileDialog()
    filedialog.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
    filedialog.setFileMode(QtWidgets.QFileDialog.Directory)
    if filedialog.exec():
        folder = filedialog.selectedFiles()

    if folder is not None:
        window = MainWindow(folder[0])
        sys.exit(app.exec_())


if __name__ == "__main__":
    run_qt_window()
