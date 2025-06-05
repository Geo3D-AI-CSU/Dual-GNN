import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import os
import trimesh


class StlToGocadConverter:
    def __init__(self, root):
        self.root = root
        self.root.title("STL至GOCAD TSurf转换器")
        self.root.geometry("600x550")
        self.root.resizable(True, True)

        self.setup_ui()

    def setup_ui(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 输入文件选择
        ttk.Label(main_frame, text="STL文件:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.input_path_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.input_path_var, width=50).grid(row=0, column=1, pady=5, padx=5)
        ttk.Button(main_frame, text="浏览...", command=self.browse_input_file).grid(row=0, column=2, pady=5)

        # 输出目录选择
        ttk.Label(main_frame, text="输出目录:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.output_dir_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.output_dir_var, width=50).grid(row=1, column=1, pady=5, padx=5)
        ttk.Button(main_frame, text="浏览...", command=self.browse_output_dir).grid(row=1, column=2, pady=5)

        # 表面名称
        ttk.Label(main_frame, text="表面名称:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.surface_name_var = tk.StringVar(value="surface")
        ttk.Entry(main_frame, textvariable=self.surface_name_var, width=50).grid(row=2, column=1, pady=5, padx=5)

        # 颜色选择
        ttk.Label(main_frame, text="表面颜色 (R G B):").grid(row=3, column=0, sticky=tk.W, pady=5)

        color_frame = ttk.Frame(main_frame)
        color_frame.grid(row=3, column=1, sticky=tk.W, pady=5)

        self.r_var = tk.DoubleVar(value=0.85)
        self.g_var = tk.DoubleVar(value=0.67)
        self.b_var = tk.DoubleVar(value=0.0)

        ttk.Label(color_frame, text="R:").pack(side=tk.LEFT)
        ttk.Spinbox(color_frame, from_=0, to=1, increment=0.1, width=5, textvariable=self.r_var).pack(side=tk.LEFT,
                                                                                                      padx=2)

        ttk.Label(color_frame, text="G:").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Spinbox(color_frame, from_=0, to=1, increment=0.1, width=5, textvariable=self.g_var).pack(side=tk.LEFT,
                                                                                                      padx=2)

        ttk.Label(color_frame, text="B:").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Spinbox(color_frame, from_=0, to=1, increment=0.1, width=5, textvariable=self.b_var).pack(side=tk.LEFT,
                                                                                                      padx=2)

        # 坐标轴映射设置
        ttk.Label(main_frame, text="坐标轴映射:", font=("", 10, "bold")).grid(row=4, column=0, columnspan=3,
                                                                              sticky=tk.W, pady=(15, 5))
        ttk.Label(main_frame, text="由于STL和GOCAD可能使用不同的坐标系，您可以调整坐标轴的映射关系:").grid(row=5,
                                                                                                          column=0,
                                                                                                          columnspan=3,
                                                                                                          sticky=tk.W,
                                                                                                          pady=(0, 10))

        # STL的X轴映射到GOCAD的哪个轴
        mapping_frame = ttk.Frame(main_frame)
        mapping_frame.grid(row=6, column=0, columnspan=3, sticky=tk.W, pady=5)

        # X轴映射
        ttk.Label(mapping_frame, text="STL的X轴 → GOCAD的:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.x_mapping = tk.StringVar(value="X")
        x_options = ttk.Combobox(mapping_frame, textvariable=self.x_mapping, values=["X", "Y", "Z", "-X", "-Y", "-Z"],
                                 width=5)
        x_options.grid(row=0, column=1, padx=5)
        x_options.current(0)

        # Y轴映射
        ttk.Label(mapping_frame, text="STL的Y轴 → GOCAD的:").grid(row=0, column=2, sticky=tk.W, padx=(20, 10))
        self.y_mapping = tk.StringVar(value="Y")
        y_options = ttk.Combobox(mapping_frame, textvariable=self.y_mapping, values=["X", "Y", "Z", "-X", "-Y", "-Z"],
                                 width=5)
        y_options.grid(row=0, column=3, padx=5)
        y_options.current(1)

        # Z轴映射
        ttk.Label(mapping_frame, text="STL的Z轴 → GOCAD的:").grid(row=0, column=4, sticky=tk.W, padx=(20, 10))
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
        filetypes = [("STL文件", "*.stl"), ("所有文件", "*.*")]
        filename = filedialog.askopenfilename(title="选择STL文件", filetypes=filetypes)
        if filename:
            self.input_path_var.set(filename)
            # 设置默认输出目录为输入文件所在目录
            if not self.output_dir_var.get():
                self.output_dir_var.set(os.path.dirname(filename))
            # 从文件名设置默认表面名称
            base_name = os.path.basename(filename)
            surface_name = os.path.splitext(base_name)[0]
            self.surface_name_var.set(surface_name)

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

    def convert(self):
        input_file = self.input_path_var.get()
        output_dir = self.output_dir_var.get()
        surface_name = self.surface_name_var.get()

        if not input_file or not output_dir:
            messagebox.showerror("错误", "请选择输入文件和输出目录")
            return

        if not surface_name:
            surface_name = "surface"

        # 验证坐标轴映射不重复
        axis_mappings = [self.x_mapping.get().replace('-', ''),
                         self.y_mapping.get().replace('-', ''),
                         self.z_mapping.get().replace('-', '')]

        if len(set(axis_mappings)) != 3:
            messagebox.showerror("错误", "坐标轴映射必须是唯一的！请检查您的设置。")
            return

        try:
            self.status_var.set("正在加载STL文件...")
            self.root.update()

            # 使用trimesh加载STL文件
            mesh = trimesh.load_mesh(input_file)

            # 获取顶点和面
            vertices = mesh.vertices
            faces = mesh.faces

            self.progress_var.set(30)
            self.status_var.set("处理网格数据...")
            self.root.update()

            # 生成输出文件名
            output_file = os.path.join(output_dir, f"{surface_name}.ts")

            # 写入GOCAD TSurf文件
            with open(output_file, 'w') as f:
                # 写入头部
                f.write(f"GOCAD TSurf 1\n")
                f.write("HEADER {\n")
                f.write(f"name: {surface_name}\n")
                f.write("mesh: on\n")
                f.write("cn: on\n")
                f.write(f"*solid*color: {self.r_var.get():.6f} {self.g_var.get():.6f} {self.b_var.get():.6f} 1\n")
                f.write("}\n")

                # 写入坐标系统
                f.write("GOCAD_ORIGINAL_COORDINATE_SYSTEM\n")
                f.write('NAME "SKUA Local"\n')
                f.write("PROJECTION Unknown\n")
                f.write("DATUM Unknown\n")
                f.write("AXIS_NAME X Y Z\n")
                f.write("AXIS_UNIT m m m\n")
                f.write("ZPOSITIVE Elevation\n")
                f.write("END_ORIGINAL_COORDINATE_SYSTEM\n")

                # 写入属性头
                f.write("PROPERTY_CLASS_HEADER X {\n")
                f.write("kind: X\n")
                f.write("unit: m\n")
                f.write("}\n")

                f.write("PROPERTY_CLASS_HEADER Y {\n")
                f.write("kind: Y\n")
                f.write("unit: m\n")
                f.write("}\n")

                f.write("PROPERTY_CLASS_HEADER Z {\n")
                f.write("kind: Z\n")
                f.write("unit: m\n")
                f.write("is_z: on\n")
                f.write("}\n")

                f.write("PROPERTY_CLASS_HEADER vector3d {\n")
                f.write("kind: Length\n")
                f.write("unit: m\n")
                f.write("}\n")

                # 写入TFACE标记
                f.write("TFACE\n")

                self.progress_var.set(50)
                self.status_var.set("正在写入顶点...")
                self.root.update()

                # 写入顶点，应用坐标映射
                for i, vertex in enumerate(vertices):
                    mapped_vertex = self.apply_coordinate_mapping(vertex)
                    f.write(
                        f"VRTX {i + 1} {mapped_vertex[0]:.15f} {mapped_vertex[1]:.15f} {mapped_vertex[2]:.15f} CNXYZ\n")
                    if i % 1000 == 0:
                        self.progress_var.set(50 + (i / len(vertices)) * 30)
                        self.root.update()

                self.progress_var.set(80)
                self.status_var.set("正在写入三角形...")
                self.root.update()

                # 写入三角形
                for i, face in enumerate(faces):
                    # 在GOCAD中，索引是从1开始的，不是从0开始
                    f.write(f"TRGL {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
                    if i % 1000 == 0:
                        self.progress_var.set(80 + (i / len(faces)) * 15)
                        self.root.update()

                # 添加一个BSTONE标记（可选，根据您的示例）
                if len(vertices) > 0:
                    f.write(f"BSTONE {1}\n")

                # 写入BORDER（可选，根据您的示例）
                # 这里简单添加一个示例边界，实际上您可能需要计算实际的边界
                if len(vertices) > 1:
                    f.write(f"BORDER {1} {1} {1}\n")

                # 写入结束标记
                f.write("END\n")

            self.progress_var.set(100)
            self.status_var.set(f"转换完成。输出保存至: {output_file}")
            messagebox.showinfo("成功", f"STL文件已成功转换为GOCAD TSurf格式。\n输出文件: {output_file}")

        except Exception as e:
            self.status_var.set(f"错误: {str(e)}")
            messagebox.showerror("错误", f"转换过程中发生错误:\n{str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = StlToGocadConverter(root)
    root.mainloop()