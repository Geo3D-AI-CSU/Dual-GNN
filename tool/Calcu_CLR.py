import sys
import os
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, Canvas, Label, Button, Frame
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import cv2
from matplotlib import path
import traceback


class ColorAnalyzer(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("图像颜色分析工具")
        self.geometry("1200x800")

        # 初始化变量
        self.image_path = None
        self.image = None
        self.cv_image = None
        self.tk_image = None
        self.selected_color = None
        self.selected_color_rgb = None
        self.polygon_points = []
        self.is_drawing = False
        self.color_tolerance = 1  # 颜色容差

        # 创建界面布局
        self.create_widgets()

    def create_widgets(self):
        # 左侧面板 - 图像和绘制区域
        self.left_panel = Frame(self, width=800, height=800)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 图像画布
        self.canvas_frame = Frame(self.left_panel)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.canvas = Canvas(self.canvas_frame, bg="lightgray", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Motion>", self.on_canvas_move)

        # 状态栏
        self.status_bar = Label(self.left_panel, text="就绪", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # 右侧面板 - 控制和信息
        self.right_panel = Frame(self, width=600, bg="#f0f0f0")
        self.right_panel.pack(side=tk.RIGHT, fill=tk.Y)

        # 控制按钮
        self.control_frame = Frame(self.right_panel, bg="#f0f0f0", padx=10, pady=10)
        self.control_frame.pack(fill=tk.X, padx=10, pady=10)

        self.load_button = Button(self.control_frame, text="加载图像", command=self.load_image)
        self.load_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.pick_color_button = Button(self.control_frame, text="选择颜色", command=self.enable_color_picking)
        self.pick_color_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.color_picking_mode = False

        self.start_polygon_button = Button(self.control_frame, text="开始绘制多边形", command=self.start_polygon)
        self.start_polygon_button.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.start_polygon_button["state"] = "disabled"

        self.end_polygon_button = Button(self.control_frame, text="结束绘制", command=self.end_polygon)
        self.end_polygon_button.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.end_polygon_button["state"] = "disabled"

        self.analyze_button = Button(self.control_frame, text="分析颜色面积", command=self.analyze_color)
        self.analyze_button.grid(row=2, columnspan=2, padx=5, pady=5, sticky="ew")
        self.analyze_button["state"] = "disabled"

        # 信息显示区域
        self.info_frame = Frame(self.right_panel, bg="#f0f0f0", padx=10, pady=10)
        self.info_frame.pack(fill=tk.X, padx=10, pady=10)

        # 选择的颜色显示
        self.color_label = Label(self.info_frame, text="选择的颜色:")
        self.color_label.grid(row=0, column=0, sticky="w", pady=5)

        self.color_display = Canvas(self.info_frame, width=50, height=25, bg="white")
        self.color_display.grid(row=0, column=1, sticky="w", pady=5, padx=5)

        self.color_value_label = Label(self.info_frame, text="RGB: 无")
        self.color_value_label.grid(row=1, columnspan=2, sticky="w", pady=2)

        # 颜色容差滑块
        self.tolerance_label = Label(self.info_frame, text=f"颜色容差: {self.color_tolerance}")
        self.tolerance_label.grid(row=2, columnspan=2, sticky="w", pady=5)

        self.tolerance_scale = tk.Scale(self.info_frame, from_=0, to=100, orient="horizontal",
                                        command=self.update_tolerance)
        self.tolerance_scale.set(self.color_tolerance)
        self.tolerance_scale.grid(row=3, columnspan=2, sticky="ew", pady=5)

        # 结果显示区域
        self.result_frame = Frame(self.right_panel, bg="#f0f0f0", padx=10, pady=10)
        self.result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.result_label = Label(self.result_frame, text="分析结果", font=("Arial", 12, "bold"))
        self.result_label.pack(anchor="w", pady=5)

        # 创建图表区域
        self.fig = Figure(figsize=(4, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.chart_canvas = FigureCanvasTkAgg(self.fig, master=self.result_frame)
        self.chart_widget = self.chart_canvas.get_tk_widget()
        self.chart_widget.pack(fill=tk.BOTH, expand=True)

        # 结果文本区域
        self.results_text = tk.Text(self.result_frame, height=8, width=40)
        self.results_text.pack(fill=tk.X, expand=True, pady=10)

    def load_image(self):
        # 打开文件对话框选择图像
        file_path = filedialog.askopenfilename(
            filetypes=[("图像文件", "*.png;*.jpg;*.jpeg;*.bmp")]
        )
        if not file_path:
            return

        self.image_path = file_path
        try:
            # 用 PIL 读取并转换为 RGB
            self.image = Image.open(file_path)
            if self.image.mode not in ('RGB', 'RGBA'):
                self.image = self.image.convert('RGB')

            # 用 OpenCV 读取（可能含透明通道）
            self.cv_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if self.cv_image is None:
                pil_img = Image.open(file_path)
                if pil_img.mode == 'RGBA':
                    pil_img = pil_img.convert('RGB')
                self.cv_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            # —— 去除透明通道 ——
            if self.cv_image.ndim == 3 and self.cv_image.shape[2] == 4:
                # BGRA -> BGR
                self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGRA2BGR)

            # 调整显示、重置状态
            self.resize_image()
            self.show_image()
            self.status_bar.config(text=f"已加载图像: {os.path.basename(file_path)}")
            self.pick_color_button["state"] = "normal"
            self.reset_selection()

        except Exception as e:
            messagebox.showerror("错误", f"无法加载图像: {e}")
            traceback.print_exc()

    def resize_image(self):
        # 获取画布大小
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # 如果画布尚未完全初始化，使用默认尺寸
        if canvas_width <= 1:
            canvas_width = 780
        if canvas_height <= 1:
            canvas_height = 580

        # 计算缩放比例
        img_width, img_height = self.image.size
        width_ratio = canvas_width / img_width
        height_ratio = canvas_height / img_height

        # 根据较小的比例进行缩放，保持宽高比
        ratio = min(width_ratio, height_ratio) * 0.9

        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)

        # 缩放图像
        self.display_image = self.image.resize((new_width, new_height), Image.LANCZOS)
        self.scale_factor = img_width / new_width  # 保存缩放比例，用于后续计算

        # 创建Tkinter兼容的图像对象
        self.tk_image = ImageTk.PhotoImage(self.display_image)

    def show_image(self):
        # 清除画布
        self.canvas.delete("all")

        # 显示图像
        self.image_id = self.canvas.create_image(
            self.canvas.winfo_width() // 2,
            self.canvas.winfo_height() // 2,
            image=self.tk_image, anchor=tk.CENTER
        )

    def enable_color_picking(self):
        # 切换颜色选择模式
        self.color_picking_mode = True
        self.is_drawing = False
        self.status_bar.config(text="请在图像上点击选择要分析的颜色")
        self.pick_color_button.config(relief=tk.SUNKEN)
        self.start_polygon_button["state"] = "disabled"
        self.end_polygon_button["state"] = "disabled"

    def on_canvas_click(self, event):
        if not self.image:
            return

        # 检查点击是否在图像内
        img_x, img_y = self.get_image_coordinates(event.x, event.y)
        if img_x is None or img_y is None:
            return

        if self.color_picking_mode:
            # 选择颜色
            self.select_color(img_x, img_y)
            self.color_picking_mode = False
            self.pick_color_button.config(relief=tk.RAISED)
            self.start_polygon_button["state"] = "normal"

        elif self.is_drawing:
            # 添加多边形点
            canvas_x, canvas_y = self.get_canvas_coordinates(img_x, img_y)
            self.polygon_points.append((img_x, img_y))

            # 绘制点
            point_id = self.canvas.create_oval(
                canvas_x - 3, canvas_y - 3,
                canvas_x + 3, canvas_y + 3,
                fill="red", outline="red", tags="polygon_point"
            )

            # 如果有多个点，绘制线段
            if len(self.polygon_points) > 1:
                prev_x, prev_y = self.get_canvas_coordinates(
                    self.polygon_points[-2][0], self.polygon_points[-2][1]
                )
                line_id = self.canvas.create_line(
                    prev_x, prev_y, canvas_x, canvas_y,
                    fill="red", width=2, tags="polygon_line"
                )

            self.status_bar.config(text=f"已添加点 {len(self.polygon_points)}，继续点击添加更多点，或点击'结束绘制'完成")

    def on_canvas_move(self, event):
        if not self.image or not self.is_drawing or len(self.polygon_points) == 0:
            return

        # 获取鼠标当前位置
        img_x, img_y = self.get_image_coordinates(event.x, event.y)
        if img_x is None or img_y is None:
            return

        # 获取最后一个点的canvas坐标
        last_x, last_y = self.get_canvas_coordinates(
            self.polygon_points[-1][0], self.polygon_points[-1][1]
        )

        # 删除之前的临时线
        self.canvas.delete("temp_line")

        # 创建新的临时线
        self.canvas.create_line(
            last_x, last_y, event.x, event.y,
            fill="red", width=2, dash=(4, 4), tags="temp_line"
        )

    def select_color(self, img_x, img_y):
        # 从 display_image 取像素（支持 RGBA/RGB/灰度）
        pixel = self.display_image.getpixel((img_x, img_y))
        if isinstance(pixel, tuple):
            if len(pixel) == 4:  # RGBA
                r, g, b, _ = pixel
            else:  # RGB
                r, g, b = pixel
        else:  # 灰度
            r = g = b = pixel

        # 映射到 cv_image 坐标
        cv_y = int(img_y * self.scale_factor)
        cv_x = int(img_x * self.scale_factor)
        if cv_y >= self.cv_image.shape[0] or cv_x >= self.cv_image.shape[1]:
            return

        # 读取 BGR
        if self.cv_image.ndim == 3 and self.cv_image.shape[2] == 3:
            b_cv, g_cv, r_cv = self.cv_image[cv_y, cv_x]
        elif self.cv_image.ndim == 2:
            v = self.cv_image[cv_y, cv_x]
            val = int(v) if not isinstance(v, np.ndarray) else int(v[0])
            b_cv = g_cv = r_cv = val
        else:  # BGRA 情况已在 load_image 去除，但保险起见仍处理
            b_cv, g_cv, r_cv, _ = self.cv_image[cv_y, cv_x]

        # 强制转 Python int，避免 uint8 溢出
        self.selected_color_rgb = (int(r_cv), int(g_cv), int(b_cv))

        # 更新 UI 显示
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        self.color_display.config(bg=hex_color)
        self.color_value_label.config(text=f"RGB: ({r_cv}, {g_cv}, {b_cv})")
        self.status_bar.config(text=f"已选择颜色 RGB: ({r_cv}, {g_cv}, {b_cv})")

    def analyze_color(self):
        if not self.selected_color_rgb or len(self.polygon_points) < 3:
            messagebox.showwarning("警告", "请先选择颜色并绘制多边形区域")
            return

        try:
            # ——— 1. 构造多边形掩码 ———
            h, w = self.cv_image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.array([
                [int(px * self.scale_factor), int(py * self.scale_factor)]
                for px, py in self.polygon_points
            ], np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [pts], 255)

            # ——— 2. 准备三通道 BGR 图像 ———
            src = self.cv_image.copy()
            if src.ndim == 3 and src.shape[2] == 4:
                src = cv2.cvtColor(src, cv2.COLOR_BGRA2BGR)
            elif src.ndim == 2:
                src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

            # ——— 3. 全图生成颜色掩码 ———
            r, g, b = self.selected_color_rgb
            tol = self.color_tolerance
            lower = np.clip([b - tol, g - tol, r - tol], 0, 255).astype(np.uint8)
            upper = np.clip([b + tol, g + tol, r + tol], 0, 255).astype(np.uint8)
            color_mask = cv2.inRange(src, lower, upper)

            # ——— 4. 计算交集像素 & 总颜色像素 ———
            intersection = cv2.bitwise_and(color_mask, mask)
            intersect_pixels = int(np.count_nonzero(intersection))
            total_color_pixels = int(np.count_nonzero(color_mask))
            ratio = intersect_pixels / total_color_pixels if total_color_pixels else 0

            # ——— 5. 文本框中显示结果 ———
            self.results_text.delete(1.0, tk.END)
            result_str = (
                f"分析结果:\n"
                f"所选颜色: RGB {self.selected_color_rgb}\n"
                f"颜色容差: {tol}\n\n"
                f"多边形内颜色像素数: {intersect_pixels}\n"
                f"全图颜色区域像素数: {total_color_pixels}\n"
                f"交集占比(交集/总色域): {ratio:.4f} ({ratio*100:.2f}%)\n"
            )
            self.results_text.insert(tk.END, result_str)

            # ——— 6. 更新柱状图 ———
            self.ax.clear()
            self.ax.bar(
                ['相交像素', '剩余颜色像素'],
                [intersect_pixels, total_color_pixels - intersect_pixels]
            )
            self.ax.set_title('多边形与颜色区域相交分析')
            self.ax.set_ylabel('像素数')
            self.fig.tight_layout()
            self.chart_canvas.draw()

            # ——— 7. 最终可视化弹窗 ———
            self.visualize_results(mask, color_mask, intersection)

        except Exception as e:
            messagebox.showerror("错误", f"分析过程中出错: {e}")
            traceback.print_exc()


    def start_polygon(self):
        if not self.selected_color_rgb:
            messagebox.showwarning("警告", "请先选择一种颜色进行分析")
            return

        self.is_drawing = True
        self.polygon_points = []
        self.canvas.delete("polygon_point", "polygon_line", "temp_line", "polygon")

        self.status_bar.config(text="请在图像上点击添加多边形的点，完成后点击'结束绘制'")
        self.start_polygon_button["state"] = "disabled"
        self.end_polygon_button["state"] = "normal"
        self.analyze_button["state"] = "disabled"

    def end_polygon(self):
        if len(self.polygon_points) < 3:
            messagebox.showwarning("警告", "请至少添加3个点来形成多边形")
            return

        self.is_drawing = False

        # 自动闭合多边形
        first_x, first_y = self.get_canvas_coordinates(
            self.polygon_points[0][0], self.polygon_points[0][1]
        )
        last_x, last_y = self.get_canvas_coordinates(
            self.polygon_points[-1][0], self.polygon_points[-1][1]
        )

        # 如果最后一个点和第一个点不同，添加闭合线
        if self.polygon_points[0] != self.polygon_points[-1]:
            self.canvas.create_line(
                last_x, last_y, first_x, first_y,
                fill="red", width=2, tags="polygon_line"
            )
            # 添加第一个点以闭合多边形
            self.polygon_points.append(self.polygon_points[0])

        # 绘制填充的多边形
        canvas_points = []
        for px, py in self.polygon_points:
            cx, cy = self.get_canvas_coordinates(px, py)
            canvas_points.extend([cx, cy])

        self.canvas.create_polygon(
            canvas_points, outline="red", fill="red",
            stipple="gray50", tags="polygon"
        )

        self.status_bar.config(text="多边形绘制完成，点击'分析颜色面积'进行分析")
        self.end_polygon_button["state"] = "disabled"
        self.analyze_button["state"] = "normal"

    def analyze_color(self):
        # 1. 检查前置条件
        if not self.selected_color_rgb or len(self.polygon_points) < 3:
            messagebox.showwarning("警告", "请先选择颜色并绘制多边形区域")
            return

        try:
            # 2. 构造多边形掩码（单通道）
            h, w = self.cv_image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.array([
                [int(px * self.scale_factor), int(py * self.scale_factor)]
                for px, py in self.polygon_points
            ], np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [pts], 255)
            polygon_pixels = int(np.count_nonzero(mask))  # 多边形区域像素数

            # 3. 准备三通道 BGR 图像
            src = self.cv_image.copy()
            if src.ndim == 3 and src.shape[2] == 4:
                src = cv2.cvtColor(src, cv2.COLOR_BGRA2BGR)
            elif src.ndim == 2:
                src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

            # 4. 全图生成颜色掩码
            r, g, b = self.selected_color_rgb
            tol = self.color_tolerance
            lower = np.clip([b - tol, g - tol, r - tol], 0, 255).astype(np.uint8)
            upper = np.clip([b + tol, g + tol, r + tol], 0, 255).astype(np.uint8)
            color_mask = cv2.inRange(src, lower, upper)
            color_pixels = int(np.count_nonzero(color_mask))  # 全图颜色像素数

            # 5. 计算交集像素
            intersection = cv2.bitwise_and(color_mask, mask)
            intersect_pixels = int(np.count_nonzero(intersection))

            # 6. 选择分母：较大的那个区域像素数
            if color_pixels >= polygon_pixels:
                denominator = color_pixels
                denom_name = "颜色区域像素数"
            else:
                denominator = polygon_pixels
                denom_name = "多边形区域像素数"

            # 7. 计算最终指标
            ratio = intersect_pixels / denominator if denominator else 0

            # 8. 文本框显示结果
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(
                tk.END,
                f"分析结果:\n"
                f"所选颜色: RGB {self.selected_color_rgb}\n"
                f"颜色容差: {tol}\n\n"
                f"多边形像素数: {polygon_pixels}\n"
                f"全图颜色像素数: {color_pixels}\n"
                f"分母（取二者中较大者——{denom_name}）: {denominator}\n"
                f"交集像素数: {intersect_pixels}\n"
                f"最终指标 (交集 / 分母): {ratio:.4f} ({ratio*100:.2f}%)\n"
            )

            # 9. 更新柱状图
            self.ax.clear()
            self.ax.bar(
                [denom_name, "交集像素"],
                [denominator, intersect_pixels]
            )
            self.ax.set_title('指标计算：交集 vs 分母')
            self.ax.set_ylabel('像素数')
            self.fig.tight_layout()
            self.chart_canvas.draw()

            # 10. 弹窗可视化
            self.visualize_results(mask, color_mask, intersection)

        except Exception as e:
            messagebox.showerror("错误", f"分析过程中出错: {e}")
            traceback.print_exc()




    def visualize_results(self, mask, color_mask, color_in_polygon):
        try:
            # 创建可视化图像
            viz_image = self.cv_image.copy()

            # 处理灰度图像情况
            if len(viz_image.shape) == 2:
                # 转换灰度图像为RGB
                viz_image = cv2.cvtColor(viz_image, cv2.COLOR_GRAY2BGR)

            # 将多边形区域半透明覆盖
            polygon_overlay = np.zeros_like(viz_image)
            polygon_overlay[mask > 0] = [0, 0, 255]  # 红色
            viz_image = cv2.addWeighted(viz_image, 1, polygon_overlay, 0.3, 0)

            # 将颜色匹配区域高亮显示
            viz_image[color_in_polygon > 0] = [0, 255, 0]  # 绿色

            # 转换为PIL格式并显示
            viz_pil = Image.fromarray(cv2.cvtColor(viz_image, cv2.COLOR_BGR2RGB))

            # 创建临时目录（如果不存在）
            temp_dir = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "temp")
            os.makedirs(temp_dir, exist_ok=True)

            # 保存临时可视化图像
            temp_viz_path = os.path.join(temp_dir, "temp_viz.png")
            viz_pil.save(temp_viz_path)

            # 打开新窗口显示结果
            result_window = tk.Toplevel(self)
            result_window.title("分析结果可视化")
            result_window.geometry("800x600")

            # 加载并调整大小显示
            viz_img = Image.open(temp_viz_path)
            w, h = viz_img.size
            if w > 800 or h > 600:
                ratio = min(800 / w, 600 / h)
                viz_img = viz_img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

            viz_tk = ImageTk.PhotoImage(viz_img)

            # 显示图像
            viz_label = Label(result_window, image=viz_tk)
            viz_label.image = viz_tk  # 保持引用
            viz_label.pack(fill=tk.BOTH, expand=True)

            # 删除临时文件
            try:
                os.remove(temp_viz_path)
            except:
                pass

        except Exception as e:
            messagebox.showerror("可视化错误", f"生成结果可视化时出错: {str(e)}")
            traceback.print_exc()

    def update_tolerance(self, value):
        self.color_tolerance = int(float(value))
        self.tolerance_label.config(text=f"颜色容差: {self.color_tolerance}")

    def get_image_coordinates(self, canvas_x, canvas_y):
        # 将画布坐标转换为图像坐标
        if not hasattr(self, 'display_image'):
            return None, None

        img_width, img_height = self.display_image.size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # 计算图像在画布中的位置
        x_offset = (canvas_width - img_width) // 2
        y_offset = (canvas_height - img_height) // 2

        # 转换坐标
        img_x = canvas_x - x_offset
        img_y = canvas_y - y_offset

        # 检查是否在图像内
        if 0 <= img_x < img_width and 0 <= img_y < img_height:
            return img_x, img_y
        else:
            return None, None

    def get_canvas_coordinates(self, img_x, img_y):
        # 将图像坐标转换为画布坐标
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_width, img_height = self.display_image.size

        # 计算图像在画布中的位置
        x_offset = (canvas_width - img_width) // 2
        y_offset = (canvas_height - img_height) // 2

        # 转换坐标
        canvas_x = img_x + x_offset
        canvas_y = img_y + y_offset

        return canvas_x, canvas_y

    def reset_selection(self):
        # 重置颜色选择和多边形
        self.selected_color_rgb = None
        self.polygon_points = []
        self.is_drawing = False
        self.color_picking_mode = False

        # 重置UI
        self.color_display.config(bg="white")
        self.color_value_label.config(text="RGB: 无")
        self.canvas.delete("polygon_point", "polygon_line", "temp_line", "polygon")

        # 重置按钮状态
        self.pick_color_button.config(relief=tk.RAISED)
        self.pick_color_button["state"] = "normal"
        self.start_polygon_button["state"] = "disabled"
        self.end_polygon_button["state"] = "disabled"
        self.analyze_button["state"] = "disabled"

        # 清除结果
        self.results_text.delete(1.0, tk.END)
        self.ax.clear()
        self.chart_canvas.draw()


if __name__ == "__main__":
    app = ColorAnalyzer()
    app.mainloop()