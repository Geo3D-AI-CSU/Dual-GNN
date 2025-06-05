import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import griddata
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import matplotlib.backends.backend_tkagg as tkagg
from matplotlib.figure import Figure
import os


class CrossSectionVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV三维剖面图生成器")
        self.root.geometry("1200x800")

        self.data = None
        self.column_names = []
        self.x_column = None
        self.y_column = None
        self.z_column = None
        self.attr_column = None
        # 插值方法和其他参数
        self.interpolation_method = "linear"  # 默认为线性插值
        self.is_continuous = True  # 默认为连续型数据

        # 创建框架
        self.left_frame = ttk.Frame(root, padding=10)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.right_frame = ttk.Frame(root, padding=10)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 左侧控制面板
        ttk.Button(self.left_frame, text="加载CSV文件", command=self.load_csv).pack(fill=tk.X, pady=5)

        # 列选择框架
        columns_frame = ttk.LabelFrame(self.left_frame, text="列选择", padding=10)
        columns_frame.pack(fill=tk.X, pady=10)

        ttk.Label(columns_frame, text="X坐标列:").grid(row=0, column=0, sticky=tk.W, pady=3)
        self.x_combobox = ttk.Combobox(columns_frame, state="readonly")
        self.x_combobox.grid(row=0, column=1, sticky=tk.W + tk.E, pady=3)
        self.x_combobox.bind("<<ComboboxSelected>>", lambda e: self.update_column_selection('x'))

        ttk.Label(columns_frame, text="Y坐标列:").grid(row=1, column=0, sticky=tk.W, pady=3)
        self.y_combobox = ttk.Combobox(columns_frame, state="readonly")
        self.y_combobox.grid(row=1, column=1, sticky=tk.W + tk.E, pady=3)
        self.y_combobox.bind("<<ComboboxSelected>>", lambda e: self.update_column_selection('y'))

        ttk.Label(columns_frame, text="Z深度列:").grid(row=2, column=0, sticky=tk.W, pady=3)
        self.z_combobox = ttk.Combobox(columns_frame, state="readonly")
        self.z_combobox.grid(row=2, column=1, sticky=tk.W + tk.E, pady=3)
        self.z_combobox.bind("<<ComboboxSelected>>", lambda e: self.update_column_selection('z'))

        ttk.Label(columns_frame, text="属性列:").grid(row=3, column=0, sticky=tk.W, pady=3)
        self.attr_combobox = ttk.Combobox(columns_frame, state="readonly")
        self.attr_combobox.grid(row=3, column=1, sticky=tk.W + tk.E, pady=3)
        self.attr_combobox.bind("<<ComboboxSelected>>", lambda e: self.update_column_selection('attr'))
        # 添加数据类型选择
        ttk.Label(columns_frame, text="数据类型:").grid(row=4, column=0, sticky=tk.W, pady=3)
        self.data_type_var = tk.StringVar(value="连续型")
        self.data_type_combobox = ttk.Combobox(columns_frame, textvariable=self.data_type_var,
                                               values=["连续型", "离散型"], state="readonly")
        self.data_type_combobox.grid(row=4, column=1, sticky=tk.W + tk.E, pady=3)
        self.data_type_combobox.bind("<<ComboboxSelected>>", self.update_data_type)
        # 剖面设置框架
        section_frame = ttk.LabelFrame(self.left_frame, text="剖面设置", padding=10)
        section_frame.pack(fill=tk.X, pady=10)

        # 第一个点的坐标
        point1_frame = ttk.LabelFrame(section_frame, text="第一个点坐标", padding=5)
        point1_frame.grid(row=0, column=0, columnspan=2, sticky=tk.W + tk.E, pady=3)

        ttk.Label(point1_frame, text="X:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.start_x = ttk.Entry(point1_frame, width=8)
        self.start_x.grid(row=0, column=1, sticky=tk.W, pady=2)

        ttk.Label(point1_frame, text="Y:").grid(row=0, column=2, sticky=tk.W, pady=2)
        self.start_y = ttk.Entry(point1_frame, width=8)
        self.start_y.grid(row=0, column=3, sticky=tk.W, pady=2)

        ttk.Label(point1_frame, text="Z:").grid(row=0, column=4, sticky=tk.W, pady=2)
        self.start_z = ttk.Entry(point1_frame, width=8)
        self.start_z.grid(row=0, column=5, sticky=tk.W, pady=2)

        # 第二个点的坐标
        point2_frame = ttk.LabelFrame(section_frame, text="第二个点坐标", padding=5)
        point2_frame.grid(row=1, column=0, columnspan=2, sticky=tk.W + tk.E, pady=3)

        ttk.Label(point2_frame, text="X:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.end_x = ttk.Entry(point2_frame, width=8)
        self.end_x.grid(row=0, column=1, sticky=tk.W, pady=2)

        ttk.Label(point2_frame, text="Y:").grid(row=0, column=2, sticky=tk.W, pady=2)
        self.end_y = ttk.Entry(point2_frame, width=8)
        self.end_y.grid(row=0, column=3, sticky=tk.W, pady=2)

        ttk.Label(point2_frame, text="Z:").grid(row=0, column=4, sticky=tk.W, pady=2)
        self.end_z = ttk.Entry(point2_frame, width=8)
        self.end_z.grid(row=0, column=5, sticky=tk.W, pady=2)

        # 其他剖面参数
        ttk.Label(section_frame, text="向下延伸深度:").grid(row=2, column=0, sticky=tk.W, pady=3)
        self.depth = ttk.Entry(section_frame)
        self.depth.grid(row=2, column=1, sticky=tk.W + tk.E, pady=3)
        self.depth.insert(0, "100")

        # 将分辨率改为像元大小
        ttk.Label(section_frame, text="像元大小(m):").grid(row=3, column=0, sticky=tk.W, pady=3)
        self.pixel_size = ttk.Entry(section_frame)
        self.pixel_size.grid(row=3, column=1, sticky=tk.W + tk.E, pady=3)
        self.pixel_size.insert(0, "1")

        # 保存设置框架
        save_frame = ttk.LabelFrame(self.left_frame, text="保存设置", padding=10)
        save_frame.pack(fill=tk.X, pady=10)

        ttk.Label(save_frame, text="DPI:").grid(row=0, column=0, sticky=tk.W, pady=3)
        self.dpi_entry = ttk.Entry(save_frame)
        self.dpi_entry.grid(row=0, column=1, sticky=tk.W + tk.E, pady=3)
        self.dpi_entry.insert(0, "300")

        # 按钮框架
        buttons_frame = ttk.Frame(self.left_frame)
        buttons_frame.pack(fill=tk.X, pady=10)

        # 生成按钮
        ttk.Button(buttons_frame, text="生成剖面图", command=self.generate_cross_section).pack(fill=tk.X, pady=5)

        # 保存按钮
        ttk.Button(buttons_frame, text="保存PNG图像", command=self.save_figure).pack(fill=tk.X, pady=5)

        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("准备就绪")
        ttk.Label(self.left_frame, textvariable=self.status_var, relief=tk.SUNKEN).pack(fill=tk.X, pady=10)

        # 右侧图形区域
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.canvas = tkagg.FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 添加工具栏
        toolbar_frame = ttk.Frame(self.right_frame)
        toolbar_frame.pack(fill=tk.X)
        toolbar = tkagg.NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()

    def update_data_type(self, event=None):
        """更新数据类型和相应的插值方法"""
        data_type = self.data_type_var.get()
        if data_type == "连续型":
            self.is_continuous = True
            self.interpolation_method = "linear"
        else:  # 离散型
            self.is_continuous = False
            self.interpolation_method = "nearest"
    def load_csv(self):
        filepath = filedialog.askopenfilename(
            title="选择CSV文件",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )

        if not filepath:
            return

        try:
            self.data = pd.read_csv(filepath)
            self.column_names = self.data.columns.tolist()

            # 更新下拉列表
            self.x_combobox['values'] = self.column_names
            self.y_combobox['values'] = self.column_names
            self.z_combobox['values'] = self.column_names
            self.attr_combobox['values'] = self.column_names

            # 尝试自动选择合适的列
            for col in self.column_names:
                col_lower = col.lower()
                if 'x' in col_lower and not self.x_column:
                    self.x_combobox.set(col)
                    self.x_column = col
                elif 'y' in col_lower and not self.y_column:
                    self.y_combobox.set(col)
                    self.y_column = col
                elif 'z' in col_lower and not self.z_column:
                    self.z_combobox.set(col)
                    self.z_column = col

            # 选择第一个数值列作为属性列
            for col in self.column_names:
                if pd.api.types.is_numeric_dtype(self.data[col]) and col not in [self.x_column, self.y_column,
                                                                                 self.z_column]:
                    self.attr_combobox.set(col)
                    self.attr_column = col
                    break

            # 更新数据范围到输入框
            if self.x_column and self.y_column and self.z_column:
                x_min, x_max = self.data[self.x_column].min(), self.data[self.x_column].max()
                y_min, y_max = self.data[self.y_column].min(), self.data[self.y_column].max()
                z_min, z_max = self.data[self.z_column].min(), self.data[self.z_column].max()

                # 设置默认的剖面线起点和终点
                self.start_x.delete(0, tk.END)
                self.start_x.insert(0, str(round(x_min + (x_max - x_min) * 0.25, 2)))

                self.start_y.delete(0, tk.END)
                self.start_y.insert(0, str(round(y_min + (y_max - y_min) * 0.25, 2)))

                self.start_z.delete(0, tk.END)
                self.start_z.insert(0, str(round(z_min + (z_max - z_min) * 0.5, 2)))

                self.end_x.delete(0, tk.END)
                self.end_x.insert(0, str(round(x_min + (x_max - x_min) * 0.75, 2)))

                self.end_y.delete(0, tk.END)
                self.end_y.insert(0, str(round(y_min + (y_max - y_min) * 0.75, 2)))

                self.end_z.delete(0, tk.END)
                self.end_z.insert(0, str(round(z_min + (z_max - z_min) * 0.5, 2)))

            self.status_var.set(f"已加载: {os.path.basename(filepath)} ({len(self.data)} 行)")
            self.plot_overview()

        except Exception as e:
            self.status_var.set(f"错误: {str(e)}")

    def update_column_selection(self, column_type):
        if column_type == 'x':
            self.x_column = self.x_combobox.get()
        elif column_type == 'y':
            self.y_column = self.y_combobox.get()
        elif column_type == 'z':
            self.z_column = self.z_combobox.get()
        elif column_type == 'attr':
            self.attr_column = self.attr_combobox.get()
        # 尝试自动判断数据类型（连续或离散）
        if self.attr_column and self.data is not None:
            unique_values = self.data[self.attr_column].nunique()
            total_values = len(self.data)

            # 如果唯一值较少或占比较低，可能是离散数据
            if unique_values < 20 or (unique_values / total_values) < 0.1:
                self.data_type_var.set("离散型")
                self.is_continuous = False
                self.interpolation_method = "nearest"
            else:
                self.data_type_var.set("连续型")
                self.is_continuous = True
                self.interpolation_method = "linear"
        # 如果所有必要列都已选择，更新概览图
        if self.x_column and self.y_column and self.z_column:
            self.plot_overview()

    def plot_overview(self):
        if not all([self.data is not None, self.x_column, self.y_column]):
            return

        self.fig.clear()
        ax = self.fig.add_subplot(111)

        # 绘制散点图，如果选择了属性列则用颜色表示
        if self.attr_column:
            scatter = ax.scatter(
                self.data[self.x_column],
                self.data[self.y_column],
                c=self.data[self.attr_column],
                cmap='viridis',
                alpha=0.7,
                s=10
            )
            self.fig.colorbar(scatter, ax=ax, label=self.attr_column)
        else:
            ax.scatter(
                self.data[self.x_column],
                self.data[self.y_column],
                alpha=0.7,
                s=10
            )

        # 如果已经输入了剖面线的起点和终点，在概览图上绘制剖面线
        try:
            start_x = float(self.start_x.get())
            start_y = float(self.start_y.get())
            end_x = float(self.end_x.get())
            end_y = float(self.end_y.get())

            ax.plot([start_x, end_x], [start_y, end_y], 'r-', linewidth=2)
            ax.plot([start_x, end_x], [start_y, end_y], 'ro', markersize=5)
            ax.text(start_x, start_y, "A", color='red', fontsize=12)
            ax.text(end_x, end_y, "B", color='red', fontsize=12)
        except:
            pass

        ax.set_xlabel(self.x_column)
        ax.set_ylabel(self.y_column)
        ax.set_title('数据概览及剖面线位置')
        ax.grid(True)

        self.canvas.draw()

    def generate_cross_section(self):
        if not all([self.data is not None, self.x_column, self.y_column, self.z_column]):
            self.status_var.set("错误: 请选择X, Y, Z列")
            return

        try:
            # 获取剖面线参数
            start_x = float(self.start_x.get())
            start_y = float(self.start_y.get())
            start_z = float(self.start_z.get())
            end_x = float(self.end_x.get())
            end_y = float(self.end_y.get())
            end_z = float(self.end_z.get())
            depth = float(self.depth.get())
            pixel_size = float(self.pixel_size.get())  # 像元大小(m)

            # 计算剖面线向量和长度
            section_vector = np.array([end_x - start_x, end_y - start_y, end_z - start_z])
            # 计算水平距离（XY平面）
            section_length = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)

            # 根据像元大小计算网格分辨率（点数）
            horizontal_resolution = int(np.ceil(section_length / pixel_size)) + 1  # +1 因为需要包括端点
            vertical_resolution = int(np.ceil(depth / pixel_size)) + 1  # +1 因为需要包括端点

            if horizontal_resolution < 2:
                horizontal_resolution = 2
            if vertical_resolution < 2:
                vertical_resolution = 2

            # 计算法向量（垂直于剖面线并且垂直于Z轴的向量）
            norm_vector = np.array([-section_vector[1], section_vector[0], 0])

            # 归一化法向量
            if np.linalg.norm(norm_vector) > 0:
                norm_vector = norm_vector / np.linalg.norm(norm_vector)
            else:
                norm_vector = np.array([1, 0, 0])  # 默认法向量

            # 创建深度向量（垂直向下）
            depth_vector = np.array([0, 0, -1])

            # 创建剖面线上的等间隔点 - 使用计算出的水平分辨率
            t = np.linspace(0, 1, horizontal_resolution)
            section_x = start_x + t * (end_x - start_x)
            section_y = start_y + t * (end_y - start_y)
            section_z = start_z + t * (end_z - start_z)

            # 创建深度网格 - 使用计算出的垂直分辨率
            depth_values = np.linspace(0, depth, vertical_resolution)

            # 创建2D网格
            section_distances = np.linspace(0, section_length, horizontal_resolution)
            section_points_x, section_points_depth = np.meshgrid(section_distances, depth_values)

            # 将网格点转换为实际坐标
            grid_x = np.zeros_like(section_points_x)
            grid_y = np.zeros_like(section_points_x)
            grid_z = np.zeros_like(section_points_x)

            for i in range(section_points_x.shape[0]):
                for j in range(section_points_x.shape[1]):
                    # 计算这个点在剖面线上的位置比例
                    ratio = section_points_x[i, j] / section_length if section_length > 0 else 0
                    if ratio <= 1:
                        # 计算剖面线上的点
                        point_on_section_x = start_x + ratio * (end_x - start_x)
                        point_on_section_y = start_y + ratio * (end_y - start_y)
                        point_on_section_z = start_z + ratio * (end_z - start_z)

                        # 根据深度向量扩展到指定深度
                        depth_ratio = section_points_depth[i, j] / depth if depth > 0 else 0

                        grid_x[i, j] = point_on_section_x
                        grid_y[i, j] = point_on_section_y
                        grid_z[i, j] = point_on_section_z + depth_ratio * depth_vector[2] * depth

            # 将网格点坐标转为扁平数组用于查询
            flat_grid_coords = np.column_stack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten()))

            # 提取数据点
            points = np.column_stack((self.data[self.x_column], self.data[self.y_column], self.data[self.z_column]))

            # 为属性值插值做准备
            from scipy.spatial import cKDTree

            if self.attr_column:
                values = self.data[self.attr_column].values

                # 根据数据类型选择插值方法
                if not self.is_continuous:
                    # 离散型数据使用最近邻插值
                    tree = cKDTree(points)
                    distances, indices = tree.query(flat_grid_coords, k=1)
                    grid_attr_flat = values[indices]
                else:
                    # 连续型数据使用线性插值
                    tree = cKDTree(points)
                    distances, indices = tree.query(flat_grid_coords, k=3)

                    # 使用反距离加权进行属性值的插值
                    weights = 1.0 / (distances + 1e-10)  # 避免除以零
                    weights = weights / np.sum(weights, axis=1)[:, np.newaxis]
                    grid_attr_flat = np.sum(weights * values[indices], axis=1)
            else:
                # 如果没有选择属性列，使用Z值
                tree = cKDTree(points)
                distances, indices = tree.query(flat_grid_coords, k=1)
                grid_attr_flat = self.data[self.z_column].values[indices]

            # 将结果重新排列成网格形状
            grid_attr = grid_attr_flat.reshape(vertical_resolution, horizontal_resolution)

            # 绘制剖面图
            self.fig.clear()
            ax = self.fig.add_subplot(111)

            # 选择合适的颜色映射
            if self.is_continuous:
                cmap = 'viridis'
                shading = 'gouraud'  # 平滑过渡
            else:
                # 获取唯一值
                if self.attr_column:
                    unique_values = np.unique(self.data[self.attr_column])
                    n_values = len(unique_values)
                    if n_values <= 20:  # 限制离散值的数量
                        cmap = plt.cm.get_cmap('tab20', n_values)
                    else:
                        cmap = 'tab20'
                else:
                    cmap = 'tab20'
                shading = 'flat'  # 平面填充，没有过渡

            # 使用pcolormesh绘制彩色网格
            if shading == 'flat':
                # 对于flat着色，C的维度需要比X和Y小1
                mesh = ax.pcolormesh(
                    section_distances,
                    depth_values,
                    grid_attr,
                    cmap=cmap,
                    shading='auto'
                )
            else:  # gouraud着色
                mesh = ax.pcolormesh(
                    section_points_x,
                    section_points_depth,
                    grid_attr,
                    cmap=cmap,
                    shading='auto'
                )

            # 添加颜色条
            cbar = self.fig.colorbar(mesh, ax=ax)
            cbar.set_label(self.attr_column if self.attr_column else self.z_column)

            # 为离散数据设置离散的颜色带刻度
            if not self.is_continuous and self.attr_column:
                unique_values = sorted(np.unique(grid_attr))
                if len(unique_values) <= 20:  # 限制显示的离散值数量
                    cbar.set_ticks(unique_values)

            # 设置轴标签和标题
            ax.set_xlabel(f"剖面线上的距离 (从点A到点B)")
            ax.set_ylabel("深度")
            ax.set_title(
                f"从 ({start_x:.2f}, {start_y:.2f}, {start_z:.2f}) 到 ({end_x:.2f}, {end_y:.2f}, {end_z:.2f}) 的剖面图")

            # 添加起点和终点标记
            ax.text(0, 0, "A", color='red', fontsize=12)
            ax.text(section_length, 0, "B", color='red', fontsize=12)

            # 反转Y轴，使深度向下
            ax.invert_yaxis()

            self.canvas.draw()
            self.status_var.set("剖面图生成完成")

        except Exception as e:
            self.status_var.set(f"错误: {str(e)}")
            import traceback
            traceback.print_exc()
    def save_figure(self):
        if not hasattr(self, 'fig'):
            self.status_var.set("错误：没有图像可保存")
            return

        try:
            # 获取DPI值
            dpi = int(self.dpi_entry.get())
            if dpi <= 0:
                messagebox.showerror("错误", "DPI必须是正整数")
                return
        except ValueError:
            messagebox.showerror("错误", "请输入有效的DPI值")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG图像", "*.png")]
        )

        if file_path:
            try:
                # 如果用户没有添加.png扩展名，确保添加
                if not file_path.lower().endswith('.png'):
                    file_path += '.png'

                self.fig.savefig(file_path, dpi=dpi, bbox_inches='tight', format='png')
                self.status_var.set(f"图像已保存至: {file_path} (DPI: {dpi})")
            except Exception as e:
                self.status_var.set(f"保存失败: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = CrossSectionVisualizer(root)
    root.mainloop()