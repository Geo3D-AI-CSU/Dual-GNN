mport sys
import os
import numpy as np
import vtk
import matplotlib.pyplot as plt
plt.rcParams['font.serif'] = ['SimSun']
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.sans-serif'] = ['Times New Roman']
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QGroupBox, QGridLayout,
                             QTabWidget, QTextEdit, QSizePolicy, QMessageBox)
from PyQt5.QtCore import Qt
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class GOCADTSReader:
    def __init__(self, filename):
        self.filename = filename
        self.vertices = []
        self.triangles = []
        self.properties = {}

    def read(self):
        with open(self.filename, 'r') as f:
            lines = f.readlines()

        current_property = None
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('VRTX'):
                parts = line.split()[1:]
                if len(parts) >= 3:
                    idx = int(parts[0]) - 1  
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])

                    while len(self.vertices) <= idx:
                        self.vertices.append(None)

                    self.vertices[idx] = (x, y, z)

            elif line.startswith('TRGL'):
                parts = line.split()[1:]
                if len(parts) >= 3:
                    v1 = int(parts[0]) - 1
                    v2 = int(parts[1]) - 1
                    v3 = int(parts[2]) - 1
                    self.triangles.append((v1, v2, v3))

            elif line.startswith('PROP_'):
                current_property = line.split()[0][5:]
                self.properties[current_property] = [None] * len(self.vertices)

            elif line.startswith('PVRTX') and current_property:
                parts = line.split()[1:]
                if len(parts) >= 4:
                    idx = int(parts[0]) - 1
                    prop_value = float(parts[-1])
                    self.properties[current_property][idx] = prop_value

        return self.vertices, self.triangles, self.properties

    def to_vtk(self):
        """Convert the parsed data into a VTK PolyData object"""
        points = vtk.vtkPoints()
        for x, y, z in self.vertices:
            points.InsertNextPoint(x, y, z)

        cells = vtk.vtkCellArray()
        for v1, v2, v3 in self.triangles:
            triangle = vtk.vtkTriangle()
            triangle.GetPointIds().SetId(0, v1)
            triangle.GetPointIds().SetId(1, v2)
            triangle.GetPointIds().SetId(2, v3)
            cells.InsertNextCell(triangle)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(cells)

        # Add attribute data
        for prop_name, prop_values in self.properties.items():
            if all(v is not None for v in prop_values):
                vtk_array = vtk.vtkDoubleArray()
                vtk_array.SetName(prop_name)
                for value in prop_values:
                    vtk_array.InsertNextValue(value)
                polydata.GetPointData().AddArray(vtk_array)

        return polydata


class SurfaceComparisonMetrics:
    """Calculate the comparative metrics between two surfaces"""

    @staticmethod
    def hausdorff_distance(surface1_points, surface2_points):
        """Calculate the Hausdorff distance"""
        tree1 = cKDTree(surface1_points)
        tree2 = cKDTree(surface2_points)

        distances1_to_2, _ = tree2.query(surface1_points)
        max_dist_1_to_2 = np.max(distances1_to_2)
        distances2_to_1, _ = tree1.query(surface2_points)
        max_dist_2_to_1 = np.max(distances2_to_1)

        hausdorff_dist = max(max_dist_1_to_2, max_dist_2_to_1)
        return hausdorff_dist

    @staticmethod
    def average_distance(surface1_points, surface2_points):
        """Calculate the average distance"""
        tree2 = cKDTree(surface2_points)
        distances1_to_2, _ = tree2.query(surface1_points)
        avg_dist = np.mean(distances1_to_2)

        return avg_dist

    @staticmethod
    def rms_distance(surface1_points, surface2_points):
        """Calculate the root mean square distance"""
        tree2 = cKDTree(surface2_points)

        distances1_to_2, _ = tree2.query(surface1_points)
        rms_dist = np.sqrt(np.mean(np.square(distances1_to_2)))

        return rms_dist

    @staticmethod
    def compute_point_distances(surface1_points, surface2_polydata):
        """Calculating point-to-surface distances using KDTree"""
        points2 = []
        for i in range(surface2_polydata.GetNumberOfPoints()):
            point = surface2_polydata.GetPoint(i)
            points2.append(point)
        points2 = np.array(points2)
        tree = cKDTree(points2)
        distances, _ = tree.query(surface1_points)

        return distances


class MatplotlibCanvas(FigureCanvas):
    """Matplotlib canvas class, for displaying charts within Qt windows"""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class MainWindow(QMainWindow):
    """Main Window Class"""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("GOCAD Isosurface Comparison Tool")
        self.setGeometry(100, 100, 1200, 800)

        # Data storage
        self.last_save_dir = None  
        self.real_surface = None
        self.predicted_surface = None
        self.real_surface_points = None
        self.predicted_surface_points = None
        self.real_filename = None
        self.predicted_filename = None
        self.last_directory = "" 
        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)

        # Left-hand control panel
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setMaximumWidth(400)

        # File Loading Group
        file_group = QGroupBox("Load file")
        file_layout = QGridLayout()

        self.load_real_btn = QPushButton("Load true isosurfaces")
        self.load_real_btn.clicked.connect(self.load_real_surface)
        self.real_label = QLabel("File not loaded")

        self.load_predicted_btn = QPushButton("Load prediction contour surfaces")
        self.load_predicted_btn.clicked.connect(self.load_predicted_surface)
        self.predicted_label = QLabel("File not loaded")

        file_layout.addWidget(self.load_real_btn, 0, 0)
        file_layout.addWidget(self.real_label, 0, 1)
        file_layout.addWidget(self.load_predicted_btn, 1, 0)
        file_layout.addWidget(self.predicted_label, 1, 1)
        file_group.setLayout(file_layout)

        # Indicator Calculation Group
        metrics_group = QGroupBox("Correspondence Index")
        metrics_layout = QVBoxLayout()

        self.calculate_btn = QPushButton("Calculate the concordance index")
        self.calculate_btn.clicked.connect(self.calculate_metrics)
        self.calculate_btn.setEnabled(False)

        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)

        metrics_layout.addWidget(self.calculate_btn)
        metrics_layout.addWidget(self.metrics_text)
        metrics_group.setLayout(metrics_layout)
        self.save_top_view_btn = QPushButton("Save top view")
        self.save_top_view_btn.clicked.connect(self.save_top_view)
        self.save_top_view_btn.setEnabled(False)
        metrics_layout.addWidget(self.calculate_btn)
        metrics_layout.addWidget(self.save_top_view_btn) 
        metrics_layout.addWidget(self.metrics_text)

        # Add to Control Panel
        control_layout.addWidget(file_group)
        control_layout.addWidget(metrics_group)
        control_layout.addStretch()

        # Right-hand display area
        display_tabs = QTabWidget()

        # 3D Visualisation Tab
        viz_widget = QWidget()
        viz_layout = QVBoxLayout(viz_widget)

        self.vtk_widget = QVTKRenderWindowInteractor()
        viz_layout.addWidget(self.vtk_widget)

        # Initialise the VTK renderer
        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()

        # Set interaction method
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)
        self.interactor.Initialize()

        # Histogram Tab
        hist_widget = QWidget()
        hist_layout = QVBoxLayout(hist_widget)

        self.hist_canvas = MatplotlibCanvas(hist_widget, width=5, height=4)
        hist_layout.addWidget(self.hist_canvas)

        # Add a tab
        display_tabs.addTab(viz_widget, "3D visualisation")
        display_tabs.addTab(hist_widget, "Distance histogram")

        # Add to the main layout
        main_layout.addWidget(control_panel, 1)
        main_layout.addWidget(display_tabs, 3)

    def load_real_surface(self):
        """Load the true isosurface file"""
        filename, _ = QFileDialog.getOpenFileName(self, "Select the true equivalent surface file", self.last_directory,"GOCAD TS file (*.ts)")
        if filename:
            try:
                reader = GOCADTSReader(filename)
                vertices, triangles, properties = reader.read()
                self.last_directory = os.path.dirname(filename)
                self.real_surface = reader.to_vtk()
                self.real_surface_points = np.array(vertices)
                self.real_filename = os.path.basename(filename)
                self.real_label.setText(self.real_filename)
                self.update_visualization()

                if self.predicted_surface is not None:
                    self.calculate_btn.setEnabled(True)
            except Exception as e:
                self.real_label.setText(f"Failed to load: {str(e)}")

    def load_predicted_surface(self):
        """Load the predicted contour file"""
        filename, _ = QFileDialog.getOpenFileName(self, "Select the true equivalent surface file", "", "GOCAD TS file (*.ts)")
        if filename:
            try:
                reader = GOCADTSReader(filename)
                vertices, triangles, properties = reader.read()
                self.predicted_surface = reader.to_vtk()
                self.predicted_surface_points = np.array(vertices)
                self.predicted_filename = os.path.basename(filename)
                self.predicted_label.setText(self.predicted_filename)
                self.update_visualization()

                if self.real_surface is not None:
                    self.calculate_btn.setEnabled(True)
            except Exception as e:
                self.predicted_label.setText(f"Failed to load: {str(e)}")

    def update_visualization(self):
        """Update 3D visualisation display"""
        self.renderer.RemoveAllViewProps()

        if self.real_surface:
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(self.real_surface)

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.0, 1.0, 0.0)  
            actor.GetProperty().SetOpacity(0.7)
            self.renderer.AddActor(actor)

        if self.predicted_surface:
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(self.predicted_surface)

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.0, 0.0, 1.0)  
            actor.GetProperty().SetOpacity(0.7)
            self.renderer.AddActor(actor)

        axes = vtk.vtkAxesActor()
        self.renderer.AddActor(axes)
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

    def calculate_metrics(self):
        """Calculate surface conformity metrics"""
        if not self.real_surface or not self.predicted_surface:
            return

        try:
            # Calculate the Hausdorff distance
            hausdorff = SurfaceComparisonMetrics.hausdorff_distance(
                self.real_surface_points, self.predicted_surface_points)

            # Average distance
            avg_dist = SurfaceComparisonMetrics.average_distance(
                self.real_surface_points, self.predicted_surface_points)

            rms_dist = SurfaceComparisonMetrics.rms_distance(
                self.real_surface_points, self.predicted_surface_points)

            distances = SurfaceComparisonMetrics.compute_point_distances(
                self.real_surface_points, self.predicted_surface)

            # Update indicator text
            metrics_text = f"Hausdorff distance: {hausdorff:.4f}\n"
            metrics_text += f"Average distance: {avg_dist:.4f}\n"
            metrics_text += f"Root mean square distance: {rms_dist:.4f}\n"
            metrics_text += f"minimum distance: {np.min(distances):.4f}\n"
            metrics_text += f"maximum distance: {np.max(distances):.4f}\n"
            metrics_text += f"Median distance: {np.median(distances):.4f}\n"
            metrics_text += f"standard deviation: {np.std(distances):.4f}\n"

            thresholds = [0.1, 0.5, 1.0, 2.0, 5.0]
            for threshold in thresholds:
                percent = 100 * np.sum(distances <= threshold) / len(distances)
                metrics_text += f"Percentage of points within ≤{threshold} distance: {percent:.2f}%\n"

            self.metrics_text.setText(metrics_text)
            self.plot_distance_histogram(distances)
            self.visualize_distance_mapping(distances)
            self.save_top_view_btn.setEnabled(True)

        except Exception as e:
            import traceback
            self.metrics_text.setText(f"Calculation failed: {str(e)}\n{traceback.format_exc()}")

    
    def save_top_view(self):
        """Generate and save the error 3D view"""
        if not self.real_surface or not self.predicted_surface:
            return

        try:
            import numpy as np
            import vtk

            distances = SurfaceComparisonMetrics.compute_point_distances(
                self.real_surface_points, self.predicted_surface)

            render_window = vtk.vtkRenderWindow()
            main_renderer = vtk.vtkRenderer()
            render_window.AddRenderer(main_renderer)
            main_renderer.SetBackground(1.0, 1.0, 1.0)

            distance_surface = vtk.vtkPolyData()
            distance_surface.DeepCopy(self.real_surface)

            distance_array = vtk.vtkDoubleArray()
            distance_array.SetName("Distance")
            for dist in distances:
                distance_array.InsertNextValue(dist)
            distance_surface.GetPointData().AddArray(distance_array)
            distance_surface.GetPointData().SetActiveScalars("Distance")

            lut = vtk.vtkLookupTable()
            lut.SetHueRange(0.667, 0.0)  
            lut.SetNumberOfColors(256)
            lut.Build()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(distance_surface)
            mapper.SetScalarRange(0, np.max(distances))
            mapper.SetLookupTable(lut)

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            main_renderer.AddActor(actor)

            scalar_bar = vtk.vtkScalarBarActor()
            scalar_bar.SetLookupTable(lut)
            # scalar_bar.SetTitle("Distance Error (m)")
            scalar_bar.SetNumberOfLabels(4)
            scalar_bar.SetLabelFormat("%.2f")
            scalar_bar.GetTitleTextProperty().SetFontSize(8)
            # scalar_bar.GetTitleTextProperty().SetBold(1)
            scalar_bar.GetTitleTextProperty().SetColor(0, 0, 0)
            scalar_bar.GetLabelTextProperty().SetFontSize(6)
            scalar_bar.GetLabelTextProperty().SetColor(0, 0, 0)
            scalar_bar.SetWidth(0.08)
            scalar_bar.SetHeight(0.4)
            scalar_bar.SetPosition(0.90, 0.3)
            main_renderer.AddActor(scalar_bar)


            camera = main_renderer.GetActiveCamera()
            camera.SetPosition(0, -1000, 1000) 
            camera.SetFocalPoint(0, 0, 0)
            camera.SetViewUp(0, 0, 1)
            main_renderer.ResetCamera()


            outline = vtk.vtkOutlineFilter()
            outline.SetInputData(distance_surface)
            outline.Update()
            outline_mapper = vtk.vtkPolyDataMapper()
            outline_mapper.SetInputConnection(outline.GetOutputPort())
            outline_actor = vtk.vtkActor()
            outline_actor.SetMapper(outline_mapper)
            outline_actor.GetProperty().SetColor(0, 0, 0)
            outline_actor.GetProperty().SetLineWidth(2)
            main_renderer.AddActor(outline_actor)

            hausdorff = SurfaceComparisonMetrics.hausdorff_distance(
                self.real_surface_points, self.predicted_surface_points)
            avg_dist = SurfaceComparisonMetrics.average_distance(
                self.real_surface_points, self.predicted_surface_points)
            rms_dist = SurfaceComparisonMetrics.rms_distance(
                self.real_surface_points, self.predicted_surface_points)

            metrics_text = vtk.vtkTextActor()
            metrics_text.SetInput(
                f"Hausdorff Distance： {hausdorff:.2f}    "
                f"Average Distance： {avg_dist:.2f}    "
                f"RMSE Distance： {rms_dist:.2f}    "
                f"Min Distance： {np.min(distances):.2f}    "
                f"Max Distance： {np.max(distances):.2f}"
            )
            text_prop = metrics_text.GetTextProperty()
            text_prop.SetFontSize(16)
            text_prop.SetBold(True)
            text_prop.SetJustificationToCentered()
            text_prop.SetColor(0, 0, 0)
            text_prop.SetFrame(1)
            text_prop.SetFrameColor(0.4, 0.4, 0.4)
            text_prop.SetBackgroundColor(1.0, 1.0, 1.0)
            text_prop.SetBackgroundOpacity(0.85)
            text_prop.SetFontFamilyToArial()
            try:
                text_prop.SetFontFamilyToFontFile()
                text_prop.SetFontFile("C:/Windows/Fonts/msyh.ttc")  
            except:
                pass 
            metrics_text.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
            metrics_text.GetPositionCoordinate().SetValue(0.5, 0.2)  
            main_renderer.AddActor(metrics_text)

            interactor = vtk.vtkRenderWindowInteractor()
            interactor.SetRenderWindow(render_window)

            compass_renderer = vtk.vtkRenderer()
            compass_renderer.SetViewport(0.04, 0.04, 0.12, 0.12)
            compass_renderer.SetLayer(1)
            compass_renderer.SetBackground(1, 1, 1)
            render_window.SetNumberOfLayers(2)
            render_window.AddRenderer(compass_renderer)

            compass_actor = vtk.vtkAxesActor()
            compass_actor.SetTotalLength(3, 3, 3)
            compass_actor.AxisLabelsOn()
            compass_actor.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetColor(0, 0, 0)
            compass_actor.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetColor(0, 0, 0)
            compass_actor.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetColor(0, 0, 0)
            compass_renderer.AddActor(compass_actor)

            compass_camera = compass_renderer.GetActiveCamera()
            compass_camera.SetPosition(1, 1, 1)
            compass_camera.SetFocalPoint(0, 0, 0)
            compass_camera.SetViewUp(0, 0, 1)
            compass_renderer.ResetCamera()

            render_window.SetSize(1200, 1000)
            render_window.Render()

            # Save image
            file_name, _ = QFileDialog.getSaveFileName(self, "Save error map", "",
                                                       "PNG Files (*.png);;JPEG Files (*.jpg)")
            if file_name:
                window_to_image_filter = vtk.vtkWindowToImageFilter()
                window_to_image_filter.SetInput(render_window)
                window_to_image_filter.SetInputBufferTypeToRGB()
                window_to_image_filter.ReadFrontBufferOff()
                window_to_image_filter.Update()

                if file_name.lower().endswith(".jpg") or file_name.lower().endswith(".jpeg"):
                    writer = vtk.vtkJPEGWriter()
                else:
                    writer = vtk.vtkPNGWriter()
                writer.SetFileName(file_name)
                writer.SetInputConnection(window_to_image_filter.GetOutputPort())
                writer.Write()

                self.last_save_dir = os.path.dirname(file_name)
                QMessageBox.information(self, "Saved successfully", f"The error map has been saved to: {file_name}")

        except Exception as e:
            import traceback
            QMessageBox.critical(self, "error", f"Failure to generate error map: {str(e)}\n{traceback.format_exc()}")
        try:
            histogram_path = os.path.join(self.last_save_dir, "distance_histogram.png")
            self.hist_canvas.fig.savefig(histogram_path, dpi=600)
        except Exception as e:
            QMessageBox.warning(self, "Failed to save histogram", f"Failed to automatically save histogram：{str(e)}")

    def plot_distance_histogram(self, distances):
        """Plot distance histogram """
        import numpy as np

        self.hist_canvas.axes.clear()
        self.hist_canvas.axes.hist(distances, bins=50, alpha=0.7, color='steelblue', edgecolor='gray')
        self.hist_canvas.axes.set_xlabel('Distance', fontsize=18)
        self.hist_canvas.axes.set_ylabel('frequency', fontsize=18)
        self.hist_canvas.axes.tick_params(axis='both', labelsize=16)

        mean_dist = np.mean(distances)
        median_dist = np.median(distances)

        self.hist_canvas.axes.axvline(mean_dist, color='r', linestyle='--', linewidth=1.5,
                                      label=f'mean: {mean_dist:.4f}')
        self.hist_canvas.axes.axvline(median_dist, color='g', linestyle='--', linewidth=1.5,
                                      label=f'median: {median_dist:.4f}')
        self.hist_canvas.axes.legend(fontsize=12)

        self.hist_canvas.fig.tight_layout()
        self.hist_canvas.draw()

    def visualize_distance_mapping(self, distances):
        """Creating a visualisation of distance mapping"""
        if not self.real_surface:
            return
        distance_surface = vtk.vtkPolyData()
        distance_surface.DeepCopy(self.real_surface)

        distance_array = vtk.vtkDoubleArray()
        distance_array.SetName("Distance")
        for dist in distances:
            distance_array.InsertNextValue(dist)

        distance_surface.GetPointData().AddArray(distance_array)
        distance_surface.GetPointData().SetActiveScalars("Distance")

        lut = vtk.vtkLookupTable()
        lut.SetHueRange(0.667, 0.0) 
        lut.SetNumberOfColors(256)
        lut.Build()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(distance_surface)
        mapper.SetScalarRange(0, np.max(distances))
        mapper.SetLookupTable(lut)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        scalar_bar = vtk.vtkScalarBarActor()
        scalar_bar.SetLookupTable(mapper.GetLookupTable())
        scalar_bar.SetTitle(" distance error(m)")
        scalar_bar.SetNumberOfLabels(4)
        scalar_bar.SetLabelFormat("%.2f")
        scalar_bar.GetTitleTextProperty().SetFontSize(8)
        scalar_bar.GetTitleTextProperty().SetBold(1)
        scalar_bar.GetLabelTextProperty().SetFontSize(8)
        scalar_bar.SetWidth(0.08) 
        scalar_bar.SetHeight(0.8)  
        scalar_bar.SetPosition(0.9, 0.1)

        self.renderer.RemoveAllViewProps()
        self.renderer.AddActor(actor)
        self.renderer.AddActor(scalar_bar)

        # Add axes
        axes = vtk.vtkAxesActor()
        self.renderer.AddActor(axes)

        # Render again
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
