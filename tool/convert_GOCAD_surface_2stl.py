import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import os
import trimesh


class GocadToStlConverter:
    def __init__(self, root):
        self.root = root
        self.root.title("GOCAD TSurf至STL转换器")
        self.root.geometry("600x550")
        self.root.resizable(True, True)

        self.setup_ui()

    def setup_ui(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 输入文件选择
        ttk.Label(main_frame, text="GOCAD TS文件:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.input_path_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.input_path_var, width=50).grid(row=0, column=1, pady=5, padx=5)
        ttk.Button(main_frame, text="浏览...", command=self.browse_input_file).grid(row=0, column=2, pady=5)

        # 输出目录选择
        ttk.Label(main_frame, text="输出目录:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.output_dir_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.output_dir_var, width=50).grid(row=1, column=1, pady=5, padx=5)
        ttk.Button(main_frame, text="浏览...", command=self.browse_output_dir).grid(row=1, column=2, pady=5)

        # STL输出文件名
        ttk.Label(main_frame, text="输出文件名:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.output_name_var = tk.StringVar(value="model")
        ttk.Entry(main_frame, textvariable=self.output_name_var, width=50).grid(row=2, column=1, pady=5, padx=5)

        # STL输出格式选择
        ttk.Label(main_frame, text="STL格式:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.format_var = tk.StringVar(value="binary")
        format_frame = ttk.Frame(main_frame)
        format_frame.grid(row=3, column=1, sticky=tk.W, pady=5)

        ttk.Radiobutton(format_frame, text="二进制 (小文件)", variable=self.format_var, value="binary").pack(
            side=tk.LEFT, padx=(0, 20))
        ttk.Radiobutton(format_frame, text="ASCII (文本)", variable=self.format_var, value="ascii").pack(side=tk.LEFT)

        # 坐标轴映射设置
        ttk.Label(main_frame, text="坐标轴映射:", font=("", 10, "bold")).grid(row=4, column=0, columnspan=3,
                                                                              sticky=tk.W, pady=(15, 5))
        ttk.Label(main_frame, text="由于GOCAD和STL可能使用不同的坐标系，您可以调整坐标轴的映射关系:").grid(row=5,
                                                                                                          column=0,
                                                                                                          columnspan=3,
                                                                                                          sticky=tk.W,
                                                                                                          pady=(0, 10))

        # 坐标轴映射框架
        mapping_frame = ttk.Frame(main_frame)
        mapping_frame.grid(row=6, column=0, columnspan=3, sticky=tk.W, pady=5)

        # GOCAD的X轴映射到STL的哪个轴
        ttk.Label(mapping_frame, text="GOCAD的X轴 → STL的:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.x_mapping = tk.StringVar(value="X")
        x_options = ttk.Combobox(mapping_frame, textvariable=self.x_mapping, values=["X", "Y", "Z", "-X", "-Y", "-Z"],
                                 width=5)
        x_options.grid(row=0, column=1, padx=5)
        x_options.current(0)

        # Y轴映射
        ttk.Label(mapping_frame, text="GOCAD的Y轴 → STL的:").grid(row=0, column=2, sticky=tk.W, padx=(20, 10))
        self.y_mapping = tk.StringVar(value="Y")
        y_options = ttk.Combobox(mapping_frame, textvariable=self.y_mapping, values=["X", "Y", "Z", "-X", "-Y", "-Z"],
                                 width=5)
        y_options.grid(row=0, column=3, padx=5)
        y_options.current(1)

        # Z轴映射
        ttk.Label(mapping_frame, text="GOCAD的Z轴 → STL的:").grid(row=0, column=4, sticky=tk.W, padx=(20, 10))
        self.z_mapping = tk.StringVar(value="Z")
        z_options = ttk.Combobox(mapping_frame, textvariable=self.z_mapping, values=["X", "Y", "Z", "-X", "-Y", "-Z"],
                                 width=5)
        z_options.grid(row=0, column=5, padx=5)
        z_options.current(2)

        # 预设按钮
        preset_frame = ttk.Frame(main_frame)
        preset_frame.grid(row=7, column=0, columnspan=3, sticky=tk.W, pady=10)

        ttk.Button(preset_frame, text="默认 (X→X, Y→Y, Z→Z)",
                   command=lambda: self.set_mapping("X", "Y", "Z")).grid(row=0, column=0, padx=5)

        ttk.Button(preset_frame, text="旋转90° (X→Y, Y→Z, Z→X)",
                   command=lambda: self.set_mapping("Y", "Z", "X")).grid(row=0, column=1, padx=5)

        ttk.Button(preset_frame, text="旋转90° (X→Z, Y→X, Z→Y)",
                   command=lambda: self.set_mapping("Z", "X", "Y")).grid(row=0, column=2, padx=5)

        # 进度条
        self.progress_var = tk.DoubleVar()
        ttk.Progressbar(main_frame, orient=tk.HORIZONTAL, length=300, mode="determinate",
                        variable=self.progress_var).grid(row=8, column=0, columnspan=3, pady=10, sticky=tk.EW)

        # 状态信息
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(main_frame, textvariable=self.status_var).grid(row=9, column=0, columnspan=3, sticky=tk.W)

        # 转换按钮
        convert_frame = ttk.Frame(main_frame)
        convert_frame.grid(row=10, column=0, columnspan=3, pady=10)

        ttk.Button(convert_frame, text="转换", command=self.convert, width=20).pack()

    def set_mapping(self, x, y, z):
        """设置预设的坐标轴映射"""
        self.x_mapping.set(x)
        self.y_mapping.set(y)
        self.z_mapping.set(z)

    def browse_input_file(self):
        filetypes = [("GOCAD TS文件", "*.ts"), ("所有文件", "*.*")]
        filename = filedialog.askopenfilename(title="选择GOCAD TS文件", filetypes=filetypes)
        if filename:
            self.input_path_var.set(filename)
            # 设置默认输出目录为输入文件所在目录
            if not self.output_dir_var.get():
                self.output_dir_var.set(os.path.dirname(filename))
            # 从文件名设置默认输出名称
            base_name = os.path.basename(filename)
            output_name = os.path.splitext(base_name)[0]
            self.output_name_var.set(output_name)

    def browse_output_dir(self):
        directory = filedialog.askdirectory(title="选择输出目录")
        if directory:
            self.output_dir_var.set(directory)

    def apply_coordinate_mapping(self, vertex):
        """应用坐标轴映射"""
        x, y, z = vertex

        # 创建映射字典
        coord_map = {'X': x, 'Y': y, 'Z': z, '-X': -x, '-Y': -y, '-Z': -z}

        # 根据用户选择的映射返回新的坐标
        new_x = coord_map[self.x_mapping.get()]
        new_y = coord_map[self.y_mapping.get()]
        new_z = coord_map[self.z_mapping.get()]

        return [new_x, new_y, new_z]

    def parse_gocad_ts(self, filepath):
        """解析GOCAD TS文件并提取顶点和三角形"""
        vertices = []
        faces = []

        with open(filepath, 'r') as f:
            lines = f.readlines()

        # 遍历每一行，寻找顶点和三角形数据
        for line in lines:
            line = line.strip()

            # 解析顶点行 (VRTX id x y z [optional])
            if line.startswith('VRTX'):
                parts = line.split()
                if len(parts) >= 5:  # 至少需要VRTX id x y z
                    # 提取顶点坐标，忽略ID和可选参数
                    x = float(parts[2])
                    y = float(parts[3])
                    z = float(parts[4])
                    vertices.append([x, y, z])

            # 解析三角形行 (TRGL v1 v2 v3)
            elif line.startswith('TRGL'):
                parts = line.split()
                if len(parts) == 4:  # 需要TRGL v1 v2 v3
                    # 在GOCAD中，索引是从1开始的，但在我们的数组中是从0开始
                    v1 = int(parts[1]) - 1
                    v2 = int(parts[2]) - 1
                    v3 = int(parts[3]) - 1
                    faces.append([v1, v2, v3])

        return np.array(vertices), np.array(faces)

    def convert(self):
        input_file = self.input_path_var.get()
        output_dir = self.output_dir_var.get()
        output_name = self.output_name_var.get()

        if not input_file or not output_dir:
            messagebox.showerror("错误", "请选择输入文件和输出目录")
            return

        if not output_name:
            output_name = "model"

        # 验证坐标轴映射不重复
        axis_mappings = [self.x_mapping.get().replace('-', ''),
                         self.y_mapping.get().replace('-', ''),
                         self.z_mapping.get().replace('-', '')]

        if len(set(axis_mappings)) != 3:
            messagebox.showerror("错误", "坐标轴映射必须是唯一的！请检查您的设置。")
            return

        try:
            self.status_var.set("正在解析GOCAD TS文件...")
            self.root.update()

            # 解析GOCAD TS文件
            vertices, faces = self.parse_gocad_ts(input_file)

            if len(vertices) == 0 or len(faces) == 0:
                messagebox.showerror("错误", "无法从TS文件中提取有效的几何数据")
                self.status_var.set("错误：无法提取几何数据")
                return

            self.progress_var.set(30)
            self.status_var.set(f"已读取 {len(vertices)} 个顶点和 {len(faces)} 个三角形...")
            self.root.update()

            # 应用坐标轴映射
            self.status_var.set("正在应用坐标映射...")
            self.root.update()

            mapped_vertices = []
            for vertex in vertices:
                mapped_vertices.append(self.apply_coordinate_mapping(vertex))

            mapped_vertices = np.array(mapped_vertices)

            self.progress_var.set(60)
            self.status_var.set("正在创建STL网格...")
            self.root.update()

            # 创建trimesh对象
            mesh = trimesh.Trimesh(vertices=mapped_vertices, faces=faces)

            # 确定输出文件路径
            stl_extension = ".stl"
            if not output_name.lower().endswith(stl_extension):
                output_name += stl_extension

            output_file = os.path.join(output_dir, output_name)

            # 保存为STL文件（修复export_stl方法调用）
            self.status_var.set("正在写入STL文件...")
            self.root.update()

            is_binary = self.format_var.get() == "binary"

            # 使用trimesh的正确导出方法
            # 方法1：直接使用export方法指定文件类型
            mesh.export(output_file, file_type='stl')

            # 如果需要指定是否为二进制格式，使用is_binary参数自行写入
            if not is_binary:  # 如果选择ASCII格式
                # 重新写入为ASCII STL
                with open(output_file, 'w') as f:
                    f.write("solid exported\n")
                    for face in mesh.faces:
                        # 获取三角形的三个顶点
                        vertices = mesh.vertices[face]
                        # 计算面法线
                        normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
                        normal = normal / np.linalg.norm(normal)

                        f.write(f"facet normal {normal[0]} {normal[1]} {normal[2]}\n")
                        f.write("  outer loop\n")
                        for vertex in vertices:
                            f.write(f"    vertex {vertex[0]} {vertex[1]} {vertex[2]}\n")
                        f.write("  endloop\n")
                        f.write("endfacet\n")
                    f.write("endsolid exported\n")

            self.progress_var.set(100)
            self.status_var.set(f"转换完成。输出保存至: {output_file}")
            messagebox.showinfo("成功", f"GOCAD TS文件已成功转换为STL格式。\n输出文件: {output_file}")

        except Exception as e:
            self.status_var.set(f"错误: {str(e)}")
            messagebox.showerror("错误", f"转换过程中发生错误:\n{str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = GocadToStlConverter(root)
    root.mainloop()