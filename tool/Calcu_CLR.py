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

        self.title("Image Colour Analysis Tool")
        self.geometry("1200x800")

        # Initialise variables
        self.image_path = None
        self.image = None
        self.cv_image = None
        self.tk_image = None
        self.selected_color = None
        self.selected_color_rgb = None
        self.polygon_points = []
        self.is_drawing = False
        self.color_tolerance = 1 

        #  Create interface layout
        self.create_widgets()

    def create_widgets(self):
        self.left_panel = Frame(self, width=800, height=800)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas_frame = Frame(self.left_panel)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas = Canvas(self.canvas_frame, bg="lightgray", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Motion>", self.on_canvas_move)

        self.status_bar = Label(self.left_panel, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.right_panel = Frame(self, width=600, bg="#f0f0f0")
        self.right_panel.pack(side=tk.RIGHT, fill=tk.Y)

        self.control_frame = Frame(self.right_panel, bg="#f0f0f0", padx=10, pady=10)
        self.control_frame.pack(fill=tk.X, padx=10, pady=10)

        self.load_button = Button(self.control_frame, text="Load image", command=self.load_image)
        self.load_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.pick_color_button = Button(self.control_frame, text="Select colour", command=self.enable_color_picking)
        self.pick_color_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.color_picking_mode = False

        self.start_polygon_button = Button(self.control_frame, text="Begin drawing the polygon", command=self.start_polygon)
        self.start_polygon_button.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.start_polygon_button["state"] = "disabled"

        self.end_polygon_button = Button(self.control_frame, text="Finish drawing", command=self.end_polygon)
        self.end_polygon_button.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.end_polygon_button["state"] = "disabled"

        self.analyze_button = Button(self.control_frame, text="Analyse colour area", command=self.analyze_color)
        self.analyze_button.grid(row=2, columnspan=2, padx=5, pady=5, sticky="ew")
        self.analyze_button["state"] = "disabled"

        # Information display area
        self.info_frame = Frame(self.right_panel, bg="#f0f0f0", padx=10, pady=10)
        self.info_frame.pack(fill=tk.X, padx=10, pady=10)

        self.color_label = Label(self.info_frame, text="Selected colour:")
        self.color_label.grid(row=0, column=0, sticky="w", pady=5)

        self.color_display = Canvas(self.info_frame, width=50, height=25, bg="white")
        self.color_display.grid(row=0, column=1, sticky="w", pady=5, padx=5)

        self.color_value_label = Label(self.info_frame, text="RGB: no")
        self.color_value_label.grid(row=1, columnspan=2, sticky="w", pady=2)

        # Colour tolerance slider
        self.tolerance_label = Label(self.info_frame, text=f"Colour tolerance: {self.color_tolerance}")
        self.tolerance_label.grid(row=2, columnspan=2, sticky="w", pady=5)

        self.tolerance_scale = tk.Scale(self.info_frame, from_=0, to=100, orient="horizontal",
                                        command=self.update_tolerance)
        self.tolerance_scale.set(self.color_tolerance)
        self.tolerance_scale.grid(row=3, columnspan=2, sticky="ew", pady=5)

        # Results display area
        self.result_frame = Frame(self.right_panel, bg="#f0f0f0", padx=10, pady=10)
        self.result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.result_label = Label(self.result_frame, text="Analysis results", font=("Arial", 12, "bold"))
        self.result_label.pack(anchor="w", pady=5)

        # Create chart area
        self.fig = Figure(figsize=(4, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.chart_canvas = FigureCanvasTkAgg(self.fig, master=self.result_frame)
        self.chart_widget = self.chart_canvas.get_tk_widget()
        self.chart_widget.pack(fill=tk.BOTH, expand=True)

        # Result text area
        self.results_text = tk.Text(self.result_frame, height=8, width=40)
        self.results_text.pack(fill=tk.X, expand=True, pady=10)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("image file", "*.png;*.jpg;*.jpeg;*.bmp")]
        )
        if not file_path:
            return

        self.image_path = file_path
        try:
            self.image = Image.open(file_path)
            if self.image.mode not in ('RGB', 'RGBA'):
                self.image = self.image.convert('RGB')

            self.cv_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if self.cv_image is None:
                pil_img = Image.open(file_path)
                if pil_img.mode == 'RGBA':
                    pil_img = pil_img.convert('RGB')
                self.cv_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


            if self.cv_image.ndim == 3 and self.cv_image.shape[2] == 4:
                # BGRA -> BGR
                self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGRA2BGR)


            self.resize_image()
            self.show_image()
            self.status_bar.config(text=f"Image loaded: {os.path.basename(file_path)}")
            self.pick_color_button["state"] = "normal"
            self.reset_selection()

        except Exception as e:
            messagebox.showerror("error", f"Image failed to load: {e}")
            traceback.print_exc()

    def resize_image(self):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()


        if canvas_width <= 1:
            canvas_width = 780
        if canvas_height <= 1:
            canvas_height = 580


        img_width, img_height = self.image.size
        width_ratio = canvas_width / img_width
        height_ratio = canvas_height / img_height


        ratio = min(width_ratio, height_ratio) * 0.9

        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)


        self.display_image = self.image.resize((new_width, new_height), Image.LANCZOS)
        self.scale_factor = img_width / new_width  
        self.tk_image = ImageTk.PhotoImage(self.display_image)

    def show_image(self):
        self.canvas.delete("all")

        self.image_id = self.canvas.create_image(
            self.canvas.winfo_width() // 2,
            self.canvas.winfo_height() // 2,
            image=self.tk_image, anchor=tk.CENTER
        )

    def enable_color_picking(self):
        self.color_picking_mode = True
        self.is_drawing = False
        self.status_bar.config(text="Please click on the image to select the colour to be analysed")
        self.pick_color_button.config(relief=tk.SUNKEN)
        self.start_polygon_button["state"] = "disabled"
        self.end_polygon_button["state"] = "disabled"

    def on_canvas_click(self, event):
        if not self.image:
            return

        img_x, img_y = self.get_image_coordinates(event.x, event.y)
        if img_x is None or img_y is None:
            return

        if self.color_picking_mode:
            self.select_color(img_x, img_y)
            self.color_picking_mode = False
            self.pick_color_button.config(relief=tk.RAISED)
            self.start_polygon_button["state"] = "normal"

        elif self.is_drawing:
            canvas_x, canvas_y = self.get_canvas_coordinates(img_x, img_y)
            self.polygon_points.append((img_x, img_y))

            point_id = self.canvas.create_oval(
                canvas_x - 3, canvas_y - 3,
                canvas_x + 3, canvas_y + 3,
                fill="red", outline="red", tags="polygon_point"
            )

            if len(self.polygon_points) > 1:
                prev_x, prev_y = self.get_canvas_coordinates(
                    self.polygon_points[-2][0], self.polygon_points[-2][1]
                )
                line_id = self.canvas.create_line(
                    prev_x, prev_y, canvas_x, canvas_y,
                    fill="red", width=2, tags="polygon_line"
                )

            self.status_bar.config(text=f"Point added {len(self.polygon_points)}，Continue clicking to add more points, or click “Finish Drawing” to complete")

    def on_canvas_move(self, event):
        if not self.image or not self.is_drawing or len(self.polygon_points) == 0:
            return

        img_x, img_y = self.get_image_coordinates(event.x, event.y)
        if img_x is None or img_y is None:
            return

        last_x, last_y = self.get_canvas_coordinates(
            self.polygon_points[-1][0], self.polygon_points[-1][1]
        )

        self.canvas.delete("temp_line")

        self.canvas.create_line(
            last_x, last_y, event.x, event.y,
            fill="red", width=2, dash=(4, 4), tags="temp_line"
        )

    def select_color(self, img_x, img_y):
        pixel = self.display_image.getpixel((img_x, img_y))
        if isinstance(pixel, tuple):
            if len(pixel) == 4:  # RGBA
                r, g, b, _ = pixel
            else:  # RGB
                r, g, b = pixel
        else: 
            r = g = b = pixel

        cv_y = int(img_y * self.scale_factor)
        cv_x = int(img_x * self.scale_factor)
        if cv_y >= self.cv_image.shape[0] or cv_x >= self.cv_image.shape[1]:
            return

        if self.cv_image.ndim == 3 and self.cv_image.shape[2] == 3:
            b_cv, g_cv, r_cv = self.cv_image[cv_y, cv_x]
        elif self.cv_image.ndim == 2:
            v = self.cv_image[cv_y, cv_x]
            val = int(v) if not isinstance(v, np.ndarray) else int(v[0])
            b_cv = g_cv = r_cv = val
        else: 
            b_cv, g_cv, r_cv, _ = self.cv_image[cv_y, cv_x]

        self.selected_color_rgb = (int(r_cv), int(g_cv), int(b_cv))

        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        self.color_display.config(bg=hex_color)
        self.color_value_label.config(text=f"RGB: ({r_cv}, {g_cv}, {b_cv})")
        self.status_bar.config(text=f"Colour selected RGB: ({r_cv}, {g_cv}, {b_cv})")

    def analyze_color(self):
        if not self.selected_color_rgb or len(self.polygon_points) < 3:
            messagebox.showwarning("Warning", "Please first select a colour and draw a polygonal area")
            return

        try:
            h, w = self.cv_image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.array([
                [int(px * self.scale_factor), int(py * self.scale_factor)]
                for px, py in self.polygon_points
            ], np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [pts], 255)

            src = self.cv_image.copy()
            if src.ndim == 3 and src.shape[2] == 4:
                src = cv2.cvtColor(src, cv2.COLOR_BGRA2BGR)
            elif src.ndim == 2:
                src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

            r, g, b = self.selected_color_rgb
            tol = self.color_tolerance
            lower = np.clip([b - tol, g - tol, r - tol], 0, 255).astype(np.uint8)
            upper = np.clip([b + tol, g + tol, r + tol], 0, 255).astype(np.uint8)
            color_mask = cv2.inRange(src, lower, upper)

            intersection = cv2.bitwise_and(color_mask, mask)
            intersect_pixels = int(np.count_nonzero(intersection))
            total_color_pixels = int(np.count_nonzero(color_mask))
            ratio = intersect_pixels / total_color_pixels if total_color_pixels else 0

            self.results_text.delete(1.0, tk.END)
            result_str = (
                f"Analysis results:\n"
                f"Selected colour: RGB {self.selected_color_rgb}\n"
                f"Colour tolerance: {tol}\n\n"
                f"Number of coloured pixels within the polygon: {intersect_pixels}\n"
                f"Number of pixels in the colour region of the entire image: {total_color_pixels}\n"
                f"Intersection proportion (intersection/total colour gamut): {ratio:.4f} ({ratio*100:.2f}%)\n"
            )
            self.results_text.insert(tk.END, result_str)

            self.ax.clear()
            self.ax.bar(
                ['intersecting pixels', 'Remaining colour pixels'],
                [intersect_pixels, total_color_pixels - intersect_pixels]
            )
            self.ax.set_title('Polygon-colour region intersection analysis')
            self.ax.set_ylabel('Number of pixels')
            self.fig.tight_layout()
            self.chart_canvas.draw()

            self.visualize_results(mask, color_mask, intersection)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during analysis: {e}")
            traceback.print_exc()


    def start_polygon(self):
        if not self.selected_color_rgb:
            messagebox.showwarning("Warning", "Please select a colour for analysis first")
            return

        self.is_drawing = True
        self.polygon_points = []
        self.canvas.delete("polygon_point", "polygon_line", "temp_line", "polygon")

        self.status_bar.config(text="Please click on the image to add polygon points. Once finished, click 'Finish Drawing'")
        self.start_polygon_button["state"] = "disabled"
        self.end_polygon_button["state"] = "normal"
        self.analyze_button["state"] = "disabled"

    def end_polygon(self):
        if len(self.polygon_points) < 3:
            messagebox.showwarning("Warning", "Please add at least three points to form a polygon.")
            return

        self.is_drawing = False

        first_x, first_y = self.get_canvas_coordinates(
            self.polygon_points[0][0], self.polygon_points[0][1]
        )
        last_x, last_y = self.get_canvas_coordinates(
            self.polygon_points[-1][0], self.polygon_points[-1][1]
        )

        if self.polygon_points[0] != self.polygon_points[-1]:
            self.canvas.create_line(
                last_x, last_y, first_x, first_y,
                fill="red", width=2, tags="polygon_line"
            )
            self.polygon_points.append(self.polygon_points[0])

        canvas_points = []
        for px, py in self.polygon_points:
            cx, cy = self.get_canvas_coordinates(px, py)
            canvas_points.extend([cx, cy])

        self.canvas.create_polygon(
            canvas_points, outline="red", fill="red",
            stipple="gray50", tags="polygon"
        )

        self.status_bar.config(text="Polygon drawing complete. Click 'Analyse Colour Area' to proceed with analysis")
        self.end_polygon_button["state"] = "disabled"
        self.analyze_button["state"] = "normal"

    def analyze_color(self):
        if not self.selected_color_rgb or len(self.polygon_points) < 3:
            messagebox.showwarning("Warning", "Please first select a colour and draw a polygonal area")
            return

        try:
            h, w = self.cv_image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.array([
                [int(px * self.scale_factor), int(py * self.scale_factor)]
                for px, py in self.polygon_points
            ], np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [pts], 255)
            polygon_pixels = int(np.count_nonzero(mask))  

            src = self.cv_image.copy()
            if src.ndim == 3 and src.shape[2] == 4:
                src = cv2.cvtColor(src, cv2.COLOR_BGRA2BGR)
            elif src.ndim == 2:
                src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

            r, g, b = self.selected_color_rgb
            tol = self.color_tolerance
            lower = np.clip([b - tol, g - tol, r - tol], 0, 255).astype(np.uint8)
            upper = np.clip([b + tol, g + tol, r + tol], 0, 255).astype(np.uint8)
            color_mask = cv2.inRange(src, lower, upper)
            color_pixels = int(np.count_nonzero(color_mask)) 

            intersection = cv2.bitwise_and(color_mask, mask)
            intersect_pixels = int(np.count_nonzero(intersection))

            if color_pixels >= polygon_pixels:
                denominator = color_pixels
                denom_name = "Number of pixels in the colour region"
            else:
                denominator = polygon_pixels
                denom_name = "Number of pixels in the polygon area"

            ratio = intersect_pixels / denominator if denominator else 0
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(
                tk.END,
                f"Analysis results:\n"
                f"Selected colour: RGB {self.selected_color_rgb}\n"
                f"Colour tolerance: {tol}\n\n"
                f"Number of pixels in a polygon: {polygon_pixels}\n"
                f"Total colour pixels in the image: {color_pixels}\n"
                f"Denominator (the larger of the two — {denom_name}): {denominator}\n"
                f"Number of pixels in the intersection: {intersect_pixels}\n"
                f"Final indicator (intersection / denominator): {ratio:.4f} ({ratio*100:.2f}%)\n"
            )
            self.ax.clear()
            self.ax.bar(
                [denom_name, "intersection pixels"],
                [denominator, intersect_pixels]
            )
            self.ax.set_title('Indicator calculation: Intersection VS Denominator')
            self.ax.set_ylabel('Number of pixels')
            self.fig.tight_layout()
            self.chart_canvas.draw()
            self.visualize_results(mask, color_mask, intersection)

        except Exception as e:
            messagebox.showerror("error", f"An error occurred during the analysis process: {e}")
            traceback.print_exc()

    def visualize_results(self, mask, color_mask, color_in_polygon):
        try:
            viz_image = self.cv_image.copy()
            if len(viz_image.shape) == 2:
                viz_image = cv2.cvtColor(viz_image, cv2.COLOR_GRAY2BGR)
            polygon_overlay = np.zeros_like(viz_image)
            polygon_overlay[mask > 0] = [0, 0, 255]  
            viz_image = cv2.addWeighted(viz_image, 1, polygon_overlay, 0.3, 0)
            viz_image[color_in_polygon > 0] = [0, 255, 0]  
            viz_pil = Image.fromarray(cv2.cvtColor(viz_image, cv2.COLOR_BGR2RGB))
            temp_dir = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "temp")
            os.makedirs(temp_dir, exist_ok=True)
            temp_viz_path = os.path.join(temp_dir, "temp_viz.png")
            viz_pil.save(temp_viz_path)
            result_window = tk.Toplevel(self)
            result_window.title("Visualisation of analysis results")
            result_window.geometry("800x600")
            viz_img = Image.open(temp_viz_path)
            w, h = viz_img.size
            if w > 800 or h > 600:
                ratio = min(800 / w, 600 / h)
                viz_img = viz_img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

            viz_tk = ImageTk.PhotoImage(viz_img)
            viz_label = Label(result_window, image=viz_tk)
            viz_label.image = viz_tk  
            viz_label.pack(fill=tk.BOTH, expand=True)
            try:
                os.remove(temp_viz_path)
            except:
                pass

        except Exception as e:
            messagebox.showerror("Visualisation error", f"An error occurred during the generation of the visualisation results: {str(e)}")
            traceback.print_exc()

    def update_tolerance(self, value):
        self.color_tolerance = int(float(value))
        self.tolerance_label.config(text=f"Colour tolerance: {self.color_tolerance}")

    def get_image_coordinates(self, canvas_x, canvas_y):
        if not hasattr(self, 'display_image'):
            return None, None

        img_width, img_height = self.display_image.size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        x_offset = (canvas_width - img_width) // 2
        y_offset = (canvas_height - img_height) // 2
        img_x = canvas_x - x_offset
        img_y = canvas_y - y_offset

        if 0 <= img_x < img_width and 0 <= img_y < img_height:
            return img_x, img_y
        else:
            return None, None

    def get_canvas_coordinates(self, img_x, img_y):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_width, img_height = self.display_image.size
        x_offset = (canvas_width - img_width) // 2
        y_offset = (canvas_height - img_height) // 2
        canvas_x = img_x + x_offset
        canvas_y = img_y + y_offset

        return canvas_x, canvas_y

    def reset_selection(self):
        self.selected_color_rgb = None
        self.polygon_points = []
        self.is_drawing = False
        self.color_picking_mode = False

        self.color_display.config(bg="white")
        self.color_value_label.config(text="RGB: no")
        self.canvas.delete("polygon_point", "polygon_line", "temp_line", "polygon")

        self.pick_color_button.config(relief=tk.RAISED)
        self.pick_color_button["state"] = "normal"
        self.start_polygon_button["state"] = "disabled"
        self.end_polygon_button["state"] = "disabled"
        self.analyze_button["state"] = "disabled"

        self.results_text.delete(1.0, tk.END)
        self.ax.clear()
        self.chart_canvas.draw()


if __name__ == "__main__":
    app = ColorAnalyzer()
    app.mainloop()
