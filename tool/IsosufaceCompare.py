mport sys
import os
import numpy as np
import vtk
import matplotlib.pyplot as plt
# 设置中文字体为黑体
plt.rcParams['font.serif'] = ['SimSun']

# 设置英文字体为 Times New Roman
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
    """读取GOCAD TS (三角网格曲面) 文件的类"""

    def __init__(self, filename):
        self.filename = filename
        self.vertices = []
        self.triangles = []
        self.properties = {}

    def read(self):
        """读取GOCAD TS文件"""
        with open(self.filename, 'r') as f:
            lines = f.readlines()

        # 解析文件
        current_property = None
        for line in lines:
            line = line.strip()

            # 跳过注释和空行
            if not line or line.startswith('#'):
                continue

            # 解析顶点
            if line.startswith('VRTX'):
                parts = line.split()[1:]
                if len(parts) >= 3:
                    idx = int(parts[0]) - 1  # GOCAD索引从1开始
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])

                    # 确保顶点列表有足够空间
                    while len(self.vertices) <= idx:
                        self.vertices.append(None)

                    self.vertices[idx] = (x, y, z)

            # 解析三角形
            elif line.startswith('TRGL'):
                parts = line.split()[1:]
                if len(parts) >= 3:
                    v1 = int(parts[0]) - 1
                    v2 = int(parts[1]) - 1
                    v3 = int(parts[2]) - 1
                    self.triangles.append((v1, v2, v3))

            # 解析属性开始
            elif line.startswith('PROP_'):
                current_property = line.split()[0][5:]  # 提取属性名称
                self.properties[current_property] = [None] * len(self.vertices)

            # 解析属性值
            elif line.startswith('PVRTX') and current_property:
                parts = line.split()[1:]
                if len(parts) >= 4:
                    idx = int(parts[0]) - 1
                    prop_value = float(parts[-1])
                    self.properties[current_property][idx] = prop_value

        return self.vertices, self.triangles, self.properties

    def to_vtk(self):
        """将解析的数据转换为VTK PolyData对象"""
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

        # 添加属性数据
        for prop_name, prop_values in self.properties.items():
            if all(v is not None for v in prop_values):
                vtk_array = vtk.vtkDoubleArray()
                vtk_array.SetName(prop_name)
                for value in prop_values:
                    vtk_array.InsertNextValue(value)
                polydata.GetPointData().AddArray(vtk_array)

        return polydata


class SurfaceComparisonMetrics:
    """计算两个表面之间的比较指标"""

    @staticmethod
    def hausdorff_distance(surface1_points, surface2_points):
        """计算Hausdorff距离"""
        tree1 = cKDTree(surface1_points)
        tree2 = cKDTree(surface2_points)

        # 计算 surface1 到 surface2 的距离
        distances1_to_2, _ = tree2.query(surface1_points)
        max_dist_1_to_2 = np.max(distances1_to_2)

        # 计算 surface2 到 surface1 的距离
        distances2_to_1, _ = tree1.query(surface2_points)
        max_dist_2_to_1 = np.max(distances2_to_1)

        # Hausdorff距离是两个最大距离中的较大者
        hausdorff_dist = max(max_dist_1_to_2, max_dist_2_to_1)
        return hausdorff_dist

    @staticmethod
    def average_distance(surface1_points, surface2_points):
        """计算平均距离"""
        tree2 = cKDTree(surface2_points)

        # 计算 surface1 到 surface2 的距离
        distances1_to_2, _ = tree2.query(surface1_points)
        avg_dist = np.mean(distances1_to_2)

        return avg_dist

    @staticmethod
    def rms_distance(surface1_points, surface2_points):
        """计算均方根距离"""
        tree2 = cKDTree(surface2_points)

        # 计算 surface1 到 surface2 的距离
        distances1_to_2, _ = tree2.query(surface1_points)
        rms_dist = np.sqrt(np.mean(np.square(distances1_to_2)))

        return rms_dist

    @staticmethod
    def compute_point_distances(surface1_points, surface2_polydata):
        """使用KDTree计算点到曲面的距离"""
        # 提取表面2的点
        points2 = []
        for i in range(surface2_polydata.GetNumberOfPoints()):
            point = surface2_polydata.GetPoint(i)
            points2.append(point)
        points2 = np.array(points2)

        # 创建KD树
        tree = cKDTree(points2)

        # 查询最近点距离
        distances, _ = tree.query(surface1_points)

        return distances


class MatplotlibCanvas(FigureCanvas):
    """Matplotlib画布类，用于在Qt窗口中显示图表"""

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
    """主窗口类"""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("GOCAD等值面比较工具")
        self.setGeometry(100, 100, 1200, 800)

        # 数据存储
        self.last_save_dir = None  # 最近保存路径，供多个方法共享
        self.real_surface = None
        self.predicted_surface = None
        self.real_surface_points = None
        self.predicted_surface_points = None
        self.real_filename = None
        self.predicted_filename = None
        self.last_directory = ""  # 存储最后使用的目录路径
        # 创建界面
        self.setup_ui()

    def setup_ui(self):
        """设置用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QHBoxLayout(central_widget)

        # 左侧控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setMaximumWidth(400)

        # 文件加载组
        file_group = QGroupBox("加载文件")
        file_layout = QGridLayout()

        self.load_real_btn = QPushButton("加载真实等值面")
        self.load_real_btn.clicked.connect(self.load_real_surface)
        self.real_label = QLabel("未加载文件")

        self.load_predicted_btn = QPushButton("加载预测等值面")
        self.load_predicted_btn.clicked.connect(self.load_predicted_surface)
        self.predicted_label = QLabel("未加载文件")

        file_layout.addWidget(self.load_real_btn, 0, 0)
        file_layout.addWidget(self.real_label, 0, 1)
        file_layout.addWidget(self.load_predicted_btn, 1, 0)
        file_layout.addWidget(self.predicted_label, 1, 1)
        file_group.setLayout(file_layout)

        # 指标计算组
        metrics_group = QGroupBox("吻合度指标")
        metrics_layout = QVBoxLayout()

        self.calculate_btn = QPushButton("计算吻合度指标")
        self.calculate_btn.clicked.connect(self.calculate_metrics)
        self.calculate_btn.setEnabled(False)

        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)

        metrics_layout.addWidget(self.calculate_btn)
        metrics_layout.addWidget(self.metrics_text)
        metrics_group.setLayout(metrics_layout)
        self.save_top_view_btn = QPushButton("保存俯视图")
        self.save_top_view_btn.clicked.connect(self.save_top_view)
        self.save_top_view_btn.setEnabled(False)
        metrics_layout.addWidget(self.calculate_btn)
        metrics_layout.addWidget(self.save_top_view_btn)  # 新增的按钮
        metrics_layout.addWidget(self.metrics_text)

        # 添加到控制面板
        control_layout.addWidget(file_group)
        control_layout.addWidget(metrics_group)
        control_layout.addStretch()

        # 右侧显示区域
        display_tabs = QTabWidget()

        # 3D可视化标签页
        viz_widget = QWidget()
        viz_layout = QVBoxLayout(viz_widget)

        self.vtk_widget = QVTKRenderWindowInteractor()
        viz_layout.addWidget(self.vtk_widget)

        # 初始化VTK渲染器
        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()

        # 设置交互方式
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)
        self.interactor.Initialize()

        # 直方图标签页
        hist_widget = QWidget()
        hist_layout = QVBoxLayout(hist_widget)

        self.hist_canvas = MatplotlibCanvas(hist_widget, width=5, height=4)
        hist_layout.addWidget(self.hist_canvas)

        # 添加标签页
        display_tabs.addTab(viz_widget, "3D可视化")
        display_tabs.addTab(hist_widget, "距离直方图")

        # 添加到主布局
        main_layout.addWidget(control_panel, 1)
        main_layout.addWidget(display_tabs, 3)

    def load_real_surface(self):
        """加载真实等值面文件"""
        filename, _ = QFileDialog.getOpenFileName(self, "选择真实等值面文件", self.last_directory,"GOCAD TS文件 (*.ts)")
        if filename:
            try:
                reader = GOCADTSReader(filename)
                vertices, triangles, properties = reader.read()
                self.last_directory = os.path.dirname(filename)
                self.real_surface = reader.to_vtk()
                self.real_surface_points = np.array(vertices)
                self.real_filename = os.path.basename(filename)
                self.real_label.setText(self.real_filename)

                # 显示
                self.update_visualization()

                # 如果两个表面都已加载，则启用计算按钮
                if self.predicted_surface is not None:
                    self.calculate_btn.setEnabled(True)
            except Exception as e:
                self.real_label.setText(f"加载失败: {str(e)}")

    def load_predicted_surface(self):
        """加载预测等值面文件"""
        filename, _ = QFileDialog.getOpenFileName(self, "选择预测等值面文件", "", "GOCAD TS文件 (*.ts)")
        if filename:
            try:
                reader = GOCADTSReader(filename)
                vertices, triangles, properties = reader.read()
                self.predicted_surface = reader.to_vtk()
                self.predicted_surface_points = np.array(vertices)
                self.predicted_filename = os.path.basename(filename)
                self.predicted_label.setText(self.predicted_filename)

                # 显示
                self.update_visualization()

                # 如果两个表面都已加载，则启用计算按钮
                if self.real_surface is not None:
                    self.calculate_btn.setEnabled(True)
            except Exception as e:
                self.predicted_label.setText(f"加载失败: {str(e)}")

    def update_visualization(self):
        """更新3D可视化显示"""
        # 清除现有的演员
        self.renderer.RemoveAllViewProps()

        # 添加真实等值面
        if self.real_surface:
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(self.real_surface)

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.0, 1.0, 0.0)  # 绿色表示真实等值面
            actor.GetProperty().SetOpacity(0.7)
            self.renderer.AddActor(actor)

        # 添加预测等值面
        if self.predicted_surface:
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(self.predicted_surface)

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.0, 0.0, 1.0)  # 蓝色表示预测等值面
            actor.GetProperty().SetOpacity(0.7)
            self.renderer.AddActor(actor)

        # 添加坐标轴
        axes = vtk.vtkAxesActor()
        self.renderer.AddActor(axes)

        # 重置相机
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

    def calculate_metrics(self):
        """计算表面吻合度指标"""
        if not self.real_surface or not self.predicted_surface:
            return

        try:
            # 计算Hausdorff距离
            hausdorff = SurfaceComparisonMetrics.hausdorff_distance(
                self.real_surface_points, self.predicted_surface_points)

            # 平均距离
            avg_dist = SurfaceComparisonMetrics.average_distance(
                self.real_surface_points, self.predicted_surface_points)

            # 均方根距离
            rms_dist = SurfaceComparisonMetrics.rms_distance(
                self.real_surface_points, self.predicted_surface_points)

            # 计算点到表面的距离 - 使用修复后的方法
            distances = SurfaceComparisonMetrics.compute_point_distances(
                self.real_surface_points, self.predicted_surface)

            # 更新指标文本
            metrics_text = f"Hausdorff距离: {hausdorff:.4f}\n"
            metrics_text += f"平均距离: {avg_dist:.4f}\n"
            metrics_text += f"均方根距离: {rms_dist:.4f}\n"

            # 计算其他统计信息
            metrics_text += f"最小距离: {np.min(distances):.4f}\n"
            metrics_text += f"最大距离: {np.max(distances):.4f}\n"
            metrics_text += f"中位数距离: {np.median(distances):.4f}\n"
            metrics_text += f"标准差: {np.std(distances):.4f}\n"

            # 计算在指定阈值内的点百分比
            thresholds = [0.1, 0.5, 1.0, 2.0, 5.0]
            for threshold in thresholds:
                percent = 100 * np.sum(distances <= threshold) / len(distances)
                metrics_text += f"距离≤{threshold}的点百分比: {percent:.2f}%\n"

            self.metrics_text.setText(metrics_text)

            # 绘制距离直方图
            self.plot_distance_histogram(distances)

            # 创建距离映射的可视化
            self.visualize_distance_mapping(distances)
            # 启用保存俯视图按钮
            self.save_top_view_btn.setEnabled(True)

        except Exception as e:
            import traceback
            self.metrics_text.setText(f"计算失败: {str(e)}\n{traceback.format_exc()}")

    # 添加新方法来保存俯视图
    def save_top_view(self):
        """生成并保存误差三维视图（优化布局：色带缩小、指标放大、指北针更靠近主图，中文支持）"""
        if not self.real_surface or not self.predicted_surface:
            return

        try:
            import numpy as np
            import vtk

            distances = SurfaceComparisonMetrics.compute_point_distances(
                self.real_surface_points, self.predicted_surface)

            # 创建窗口和主渲染器
            render_window = vtk.vtkRenderWindow()
            main_renderer = vtk.vtkRenderer()
            render_window.AddRenderer(main_renderer)
            main_renderer.SetBackground(1.0, 1.0, 1.0)

            # 构建误差面
            distance_surface = vtk.vtkPolyData()
            distance_surface.DeepCopy(self.real_surface)

            distance_array = vtk.vtkDoubleArray()
            distance_array.SetName("Distance")
            for dist in distances:
                distance_array.InsertNextValue(dist)
            distance_surface.GetPointData().AddArray(distance_array)
            distance_surface.GetPointData().SetActiveScalars("Distance")

            # 设置颜色映射
            lut = vtk.vtkLookupTable()
            lut.SetHueRange(0.667, 0.0)  # 蓝到红
            lut.SetNumberOfColors(256)
            lut.Build()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(distance_surface)
            mapper.SetScalarRange(0, np.max(distances))
            mapper.SetLookupTable(lut)

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            main_renderer.AddActor(actor)

            # 缩小的色带，靠近主图
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

            # 设置相机从南方45°俯视
            camera = main_renderer.GetActiveCamera()
            camera.SetPosition(0, -1000, 1000)  # 南方向北
            camera.SetFocalPoint(0, 0, 0)
            camera.SetViewUp(0, 0, 1)
            main_renderer.ResetCamera()

            # 添加外框线
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

            # 计算指标值
            hausdorff = SurfaceComparisonMetrics.hausdorff_distance(
                self.real_surface_points, self.predicted_surface_points)
            avg_dist = SurfaceComparisonMetrics.average_distance(
                self.real_surface_points, self.predicted_surface_points)
            rms_dist = SurfaceComparisonMetrics.rms_distance(
                self.real_surface_points, self.predicted_surface_points)

            # 美观放大的指标框，支持中文显示
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
            # 设置支持中文的字体（Windows下为“微软雅黑”）
            text_prop.SetFontFamilyToArial()
            try:
                text_prop.SetFontFamilyToFontFile()
                text_prop.SetFontFile("C:/Windows/Fonts/msyh.ttc")  # 微软雅黑字体路径
            except:
                pass  # 如果路径无效则保持默认字体
            metrics_text.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
            metrics_text.GetPositionCoordinate().SetValue(0.5, 0.2)  # 中下部，贴近主图
            main_renderer.AddActor(metrics_text)

            # 创建渲染器交互器
            interactor = vtk.vtkRenderWindowInteractor()
            interactor.SetRenderWindow(render_window)

            # 左下角三维方向指北针（放大+靠近主图）
            compass_renderer = vtk.vtkRenderer()
            compass_renderer.SetViewport(0.04, 0.04, 0.12, 0.12)
            compass_renderer.SetLayer(1)
            compass_renderer.SetBackground(1, 1, 1)
            render_window.SetNumberOfLayers(2)
            render_window.AddRenderer(compass_renderer)

            compass_actor = vtk.vtkAxesActor()
            compass_actor.SetTotalLength(3, 3, 3)  # 放大
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

            # 渲染窗口设置
            render_window.SetSize(1200, 1000)
            render_window.Render()

            # 保存图像
            file_name, _ = QFileDialog.getSaveFileName(self, "保存误差图", "",
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
                QMessageBox.information(self, "保存成功", f"误差图已保存到: {file_name}")

        except Exception as e:
            import traceback
            QMessageBox.critical(self, "错误", f"生成误差图失败: {str(e)}\n{traceback.format_exc()}")
        # 自动保存直方图
        try:
            histogram_path = os.path.join(self.last_save_dir, "distance_histogram.png")
            self.hist_canvas.fig.savefig(histogram_path, dpi=600)
        except Exception as e:
            QMessageBox.warning(self, "直方图保存失败", f"自动保存直方图失败：{str(e)}")

    def plot_distance_histogram(self, distances):
        """绘制距离直方图（不保存，仅更新界面）"""
        import numpy as np

        self.hist_canvas.axes.clear()
        self.hist_canvas.axes.hist(distances, bins=50, alpha=0.7, color='steelblue', edgecolor='gray')
        self.hist_canvas.axes.set_xlabel('距离', fontsize=18)
        self.hist_canvas.axes.set_ylabel('频率', fontsize=18)
        self.hist_canvas.axes.tick_params(axis='both', labelsize=16)

        mean_dist = np.mean(distances)
        median_dist = np.median(distances)

        self.hist_canvas.axes.axvline(mean_dist, color='r', linestyle='--', linewidth=1.5,
                                      label=f'平均值: {mean_dist:.4f}')
        self.hist_canvas.axes.axvline(median_dist, color='g', linestyle='--', linewidth=1.5,
                                      label=f'中位数: {median_dist:.4f}')
        self.hist_canvas.axes.legend(fontsize=12)

        self.hist_canvas.fig.tight_layout()
        self.hist_canvas.draw()

    def visualize_distance_mapping(self, distances):
        """创建距离映射的可视化"""
        if not self.real_surface:
            return

        # 复制一个表面用于可视化
        distance_surface = vtk.vtkPolyData()
        distance_surface.DeepCopy(self.real_surface)

        # 创建距离标量数组
        distance_array = vtk.vtkDoubleArray()
        distance_array.SetName("Distance")
        for dist in distances:
            distance_array.InsertNextValue(dist)

        # 将数组添加到点数据
        distance_surface.GetPointData().AddArray(distance_array)
        distance_surface.GetPointData().SetActiveScalars("Distance")

        # 设置颜色映射
        lut = vtk.vtkLookupTable()
        lut.SetHueRange(0.667, 0.0)  # 从蓝色到红色
        lut.SetNumberOfColors(256)
        lut.Build()

        # 创建映射器
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(distance_surface)
        mapper.SetScalarRange(0, np.max(distances))
        mapper.SetLookupTable(lut)

        # 创建演员
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # 添加改进的颜色条
        scalar_bar = vtk.vtkScalarBarActor()
        scalar_bar.SetLookupTable(mapper.GetLookupTable())
        scalar_bar.SetTitle(" distance error(m)")
        scalar_bar.SetNumberOfLabels(4)
        scalar_bar.SetLabelFormat("%.2f")
        scalar_bar.GetTitleTextProperty().SetFontSize(8)
        scalar_bar.GetTitleTextProperty().SetBold(1)
        scalar_bar.GetLabelTextProperty().SetFontSize(8)
        scalar_bar.SetWidth(0.08)  # 调整宽度
        scalar_bar.SetHeight(0.8)  # 调整高度
        scalar_bar.SetPosition(0.9, 0.1)  # 调整位置

        # 清除现有的演员并添加新的
        self.renderer.RemoveAllViewProps()
        self.renderer.AddActor(actor)
        self.renderer.AddActor(scalar_bar)

        # 添加坐标轴
        axes = vtk.vtkAxesActor()
        self.renderer.AddActor(axes)

        # 重新渲染
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())