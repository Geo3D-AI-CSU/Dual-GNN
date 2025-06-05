import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import pyvista as pv
import os
from tkinter import ttk


def load_and_delaunay(csv_file, x_col, y_col, z_col, value_col):
    """
    从不规则点云生成四面体网格，并插入属性值
    """
    df = pd.read_csv(csv_file)
    points = df[[x_col, y_col, z_col]].values
    values = df[value_col].values

    pdata = pv.PolyData(points)
    pdata.point_data[value_col] = values

    # 使用 Delaunay 3D 生成三维四面体剖分
    tetra = pdata.delaunay_3d()
    return tetra


def extract_and_save_isosurfaces(mesh, value_col, isovalues, output_dir, visualize=False, progress_var=None,
                                 smoothing_iterations=100, smoothing_relaxation=0.2):
    """
    提取并保存等值面，添加平滑处理

    参数:
    - smoothing_iterations: 平滑迭代次数
    - smoothing_relaxation: 平滑松弛因子 (0-1之间)
    """
    os.makedirs(output_dir, exist_ok=True)
    plotter = pv.Plotter()

    total = len(isovalues)
    for i, val in enumerate(isovalues):
        print(f"⛏️ 正在提取等值面: {val}")

        # 提取等值面
        surf = mesh.contour(isosurfaces=[val], scalars=value_col)

        # 应用平滑处理
        surf_smoothed = surf.smooth(n_iter=smoothing_iterations, relaxation_factor=smoothing_relaxation)

        if visualize:
            plotter.add_mesh(surf_smoothed, label=f"Isovalue {val}", show_edges=False)

        out_path = os.path.join(output_dir, f"isosurface_{val}.stl")
        surf_smoothed.save(out_path)
        print(f"✅ 已保存: {out_path}")

        # 更新进度条
        if progress_var:
            progress_var.set((i + 1) / total * 100)

    if visualize:
        plotter.add_legend()
        plotter.show()


def run_gui():
    root = tk.Tk()
    root.title("不规则点云等值面提取")
    root.geometry("600x550")
    state = {}

    def choose_file():
        path = filedialog.askopenfilename(filetypes=[("CSV 文件", "*.csv")])
        file_entry.delete(0, tk.END)
        file_entry.insert(0, path)
        state['csv_path'] = path
        try:
            df = pd.read_csv(path, nrows=5)
            cols = df.columns.tolist()
            for menu in (x_menu, y_menu, z_menu, val_menu):
                menu['menu'].delete(0, 'end')
            for col in cols:
                for var, menu in [(x_var, x_menu), (y_var, y_menu), (z_var, z_menu), (val_var, val_menu)]:
                    menu['menu'].add_command(label=col, command=tk._setit(var, col))
        except Exception as e:
            messagebox.showerror("读取失败", str(e))

    def run_process():
        try:
            csv_path = file_entry.get()
            x_col = x_var.get()
            y_col = y_var.get()
            z_col = z_var.get()
            value_col = val_var.get()
            isovalues = [float(v.strip()) for v in iso_entry.get().split(",")]
            output_dir = filedialog.askdirectory(title="选择输出目录")

            # 获取平滑参数
            smoothing_iterations = int(smooth_iter_entry.get())
            smoothing_relaxation = float(smooth_relax_entry.get())

            mesh = load_and_delaunay(csv_path, x_col, y_col, z_col, value_col)
            extract_and_save_isosurfaces(
                mesh, value_col, isovalues,
                output_dir, visualize=True,
                progress_var=progress_var,
                smoothing_iterations=smoothing_iterations,
                smoothing_relaxation=smoothing_relaxation
            )

            messagebox.showinfo("完成", f"等值面已保存至：{output_dir}")
        except Exception as e:
            messagebox.showerror("出错了", str(e))

    # 创建基本UI框架
    input_frame = ttk.LabelFrame(root, text="数据输入")
    input_frame.pack(fill="both", expand=True, padx=10, pady=5)

    ttk.Label(input_frame, text="CSV文件路径：").pack(anchor="w", padx=5, pady=2)
    file_entry = ttk.Entry(input_frame, width=60)
    file_entry.pack(padx=5, pady=2, fill="x")
    ttk.Button(input_frame, text="浏览", command=choose_file).pack(padx=5, pady=2)

    # 创建列选择框架
    columns_frame = ttk.Frame(input_frame)
    columns_frame.pack(fill="x", padx=5, pady=5)

    column_labels = ["X列名", "Y列名", "Z列名", "属性列名"]
    var_names = ['x', 'y', 'z', 'val']

    for i, (label, varname) in enumerate(zip(column_labels, var_names)):
        ttk.Label(columns_frame, text=label).grid(row=0, column=i, padx=5)
        v = tk.StringVar()
        menu = ttk.OptionMenu(columns_frame, v, "")
        menu.grid(row=1, column=i, padx=5, pady=2)
        state[f"{varname}_var"] = v
        state[f"{varname}_menu"] = menu

    x_var, y_var, z_var, val_var = state['x_var'], state['y_var'], state['z_var'], state['val_var']
    x_menu, y_menu, z_menu, val_menu = state['x_menu'], state['y_menu'], state['z_menu'], state['val_menu']

    # 创建参数设置框架
    param_frame = ttk.LabelFrame(root, text="等值面参数")
    param_frame.pack(fill="both", expand=True, padx=10, pady=5)

    ttk.Label(param_frame, text="等值面值（逗号分隔）").pack(anchor="w", padx=5, pady=2)
    iso_entry = ttk.Entry(param_frame)
    iso_entry.insert(0, "20, 30, 50")
    iso_entry.pack(padx=5, pady=2, fill="x")

    # 添加平滑参数
    smooth_frame = ttk.Frame(param_frame)
    smooth_frame.pack(fill="x", padx=5, pady=5)

    ttk.Label(smooth_frame, text="平滑迭代次数:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
    smooth_iter_entry = ttk.Entry(smooth_frame, width=10)
    smooth_iter_entry.insert(0, "100")
    smooth_iter_entry.grid(row=0, column=1, padx=5, pady=2)

    ttk.Label(smooth_frame, text="平滑松弛因子(0-1):").grid(row=1, column=0, padx=5, pady=2, sticky="w")
    smooth_relax_entry = ttk.Entry(smooth_frame, width=10)
    smooth_relax_entry.insert(0, "0.2")
    smooth_relax_entry.grid(row=1, column=1, padx=5, pady=2)

    # 添加帮助提示
    ttk.Label(param_frame, text="提示: 平滑迭代次数越高、松弛因子越大，等值面越平滑，但计算时间会更长").pack(anchor="w",
                                                                                                          padx=5,
                                                                                                          pady=2)

    # 执行按钮
    ttk.Button(root, text="提取等值面并保存", command=run_process, style="Accent.TButton").pack(pady=10)

    # 创建进度条
    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=100)
    progress_bar.pack(fill='x', padx=10, pady=10)

    # 设置样式
    style = ttk.Style()
    style.configure("Accent.TButton", font=("Arial", 10, "bold"))

    root.mainloop()


if __name__ == "__main__":
    run_gui()