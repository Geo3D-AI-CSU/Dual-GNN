import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import os
import trimesh


class StlToGocadConverter:
    def __init__(self, root):
        self.root = root
        self.root.title("STL to GOCAD TSurf Converter")
        self.root.geometry("600x550")
        self.root.resizable(True, True)

        self.setup_ui()

    def setup_ui(self):
        # Create the main framework
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Input file selection
        ttk.Label(main_frame, text="STL File:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.input_path_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.input_path_var, width=50).grid(row=0, column=1, pady=5, padx=5)
        ttk.Button(main_frame, text="Browse...", command=self.browse_input_file).grid(row=0, column=2, pady=5)

        # Output directory selection
        ttk.Label(main_frame, text="Output directory:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.output_dir_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.output_dir_var, width=50).grid(row=1, column=1, pady=5, padx=5)
        ttk.Button(main_frame, text="Browse...", command=self.browse_output_dir).grid(row=1, column=2, pady=5)

        # Surface designation
        ttk.Label(main_frame, text="Surface designation:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.surface_name_var = tk.StringVar(value="surface")
        ttk.Entry(main_frame, textvariable=self.surface_name_var, width=50).grid(row=2, column=1, pady=5, padx=5)

        # Colour selection
        ttk.Label(main_frame, text="Surface colour (R G B):").grid(row=3, column=0, sticky=tk.W, pady=5)

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

        # Coordinate Axis Mapping Settings
        ttk.Label(main_frame, text="Coordinate axis mapping:", font=("", 10, "bold")).grid(row=4, column=0, columnspan=3,
                                                                              sticky=tk.W, pady=(15, 5))
        ttk.Label(main_frame, text="As STL and GOCAD may employ different coordinate systems, you may adjust the mapping relationship between the coordinate axes.:").grid(row=5,
                                                                                                          column=0,
                                                                                                          columnspan=3,
                                                                                                          sticky=tk.W,
                                                                                                          pady=(0, 10))

        # Which axis in GOCAD corresponds to the X-axis in STL?
        mapping_frame = ttk.Frame(main_frame)
        mapping_frame.grid(row=6, column=0, columnspan=3, sticky=tk.W, pady=5)

        # X-axis mapping
        ttk.Label(mapping_frame, text="STL → GOCAD:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.x_mapping = tk.StringVar(value="X")
        x_options = ttk.Combobox(mapping_frame, textvariable=self.x_mapping, values=["X", "Y", "Z", "-X", "-Y", "-Z"],
                                 width=5)
        x_options.grid(row=0, column=1, padx=5)
        x_options.current(0)

        # Y-axis mapping
        ttk.Label(mapping_frame, text="STL → GOCAD:").grid(row=0, column=2, sticky=tk.W, padx=(20, 10))
        self.y_mapping = tk.StringVar(value="Y")
        y_options = ttk.Combobox(mapping_frame, textvariable=self.y_mapping, values=["X", "Y", "Z", "-X", "-Y", "-Z"],
                                 width=5)
        y_options.grid(row=0, column=3, padx=5)
        y_options.current(1)

        # Z-axis mapping
        ttk.Label(mapping_frame, text="STL → GOCAD:").grid(row=0, column=4, sticky=tk.W, padx=(20, 10))
        self.z_mapping = tk.StringVar(value="Z")
        z_options = ttk.Combobox(mapping_frame, textvariable=self.z_mapping, values=["X", "Y", "Z", "-X", "-Y", "-Z"],
                                 width=5)
        z_options.grid(row=0, column=5, padx=5)
        z_options.current(2)

        # Preset button
        preset_frame = ttk.Frame(main_frame)
        preset_frame.grid(row=7, column=0, columnspan=3, sticky=tk.W, pady=10)

        ttk.Button(preset_frame, text="Default (X→X, Y→Y, Z→Z)",
                   command=lambda: self.set_mapping("X", "Y", "Z")).grid(row=0, column=0, padx=5)

        ttk.Button(preset_frame, text="Rotation 90° (X→Y, Y→Z, Z→X)",
                   command=lambda: self.set_mapping("Y", "Z", "X")).grid(row=0, column=1, padx=5)

        ttk.Button(preset_frame, text="Rotation 90° (X→Z, Y→X, Z→Y)",
                   command=lambda: self.set_mapping("Z", "X", "Y")).grid(row=0, column=2, padx=5)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        ttk.Progressbar(main_frame, orient=tk.HORIZONTAL, length=300, mode="determinate",
                        variable=self.progress_var).grid(row=8, column=0, columnspan=3, pady=10, sticky=tk.EW)

        # Status information
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(main_frame, textvariable=self.status_var).grid(row=9, column=0, columnspan=3, sticky=tk.W)

        # Convert button
        convert_frame = ttk.Frame(main_frame)
        convert_frame.grid(row=10, column=0, columnspan=3, pady=10)

        ttk.Button(convert_frame, text="Convert", command=self.convert, width=20).pack()

    def set_mapping(self, x, y, z):
        """Set the preset axis mapping"""
        self.x_mapping.set(x)
        self.y_mapping.set(y)
        self.z_mapping.set(z)

    def browse_input_file(self):
        filetypes = [("STL File", "*.stl"), ("All of File", "*.*")]
        filename = filedialog.askopenfilename(title="Select STL file", filetypes=filetypes)
        if filename:
            self.input_path_var.set(filename)
            # Set the default output directory to the directory containing the input files
            if not self.output_dir_var.get():
                self.output_dir_var.set(os.path.dirname(filename))
            # Set the default surface name from the file name
            base_name = os.path.basename(filename)
            surface_name = os.path.splitext(base_name)[0]
            self.surface_name_var.set(surface_name)

    def browse_output_dir(self):
        directory = filedialog.askdirectory(title="Select output directory")
        if directory:
            self.output_dir_var.set(directory)

    def apply_coordinate_mapping(self, vertex):
        """Application of coordinate axis mapping"""
        x, y, z = vertex

        # Create a mapping dictionary
        coord_map = {'X': x, 'Y': y, 'Z': z, '-X': -x, '-Y': -y, '-Z': -z}

        new_x = coord_map[self.x_mapping.get()]
        new_y = coord_map[self.y_mapping.get()]
        new_z = coord_map[self.z_mapping.get()]

        return [new_x, new_y, new_z]

    def convert(self):
        input_file = self.input_path_var.get()
        output_dir = self.output_dir_var.get()
        surface_name = self.surface_name_var.get()

        if not input_file or not output_dir:
            messagebox.showerror("Error", "Please select the input file and output directory")
            return

        if not surface_name:
            surface_name = "surface"

        axis_mappings = [self.x_mapping.get().replace('-', ''),
                         self.y_mapping.get().replace('-', ''),
                         self.z_mapping.get().replace('-', '')]

        if len(set(axis_mappings)) != 3:
            messagebox.showerror("Error", "Coordinate axis mappings must be unique! Please check your settings")
            return

        try:
            self.status_var.set("Loading STL file...")
            self.root.update()
            mesh = trimesh.load_mesh(input_file)
            vertices = mesh.vertices
            faces = mesh.faces

            self.progress_var.set(30)
            self.status_var.set("Processing grid data...")
            self.root.update()

            output_file = os.path.join(output_dir, f"{surface_name}.ts")

            # Write to GOCAD TSurf file
            with open(output_file, 'w') as f:

                f.write(f"GOCAD TSurf 1\n")
                f.write("HEADER {\n")
                f.write(f"name: {surface_name}\n")
                f.write("mesh: on\n")
                f.write("cn: on\n")
                f.write(f"*solid*color: {self.r_var.get():.6f} {self.g_var.get():.6f} {self.b_var.get():.6f} 1\n")
                f.write("}\n")
                f.write("GOCAD_ORIGINAL_COORDINATE_SYSTEM\n")
                f.write('NAME "SKUA Local"\n')
                f.write("PROJECTION Unknown\n")
                f.write("DATUM Unknown\n")
                f.write("AXIS_NAME X Y Z\n")
                f.write("AXIS_UNIT m m m\n")
                f.write("ZPOSITIVE Elevation\n")
                f.write("END_ORIGINAL_COORDINATE_SYSTEM\n")
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
                f.write("TFACE\n")

                self.progress_var.set(50)
                self.status_var.set("Writing vertices...")
                self.root.update()

                # Write vertices, apply coordinate mapping
                for i, vertex in enumerate(vertices):
                    mapped_vertex = self.apply_coordinate_mapping(vertex)
                    f.write(
                        f"VRTX {i + 1} {mapped_vertex[0]:.15f} {mapped_vertex[1]:.15f} {mapped_vertex[2]:.15f} CNXYZ\n")
                    if i % 1000 == 0:
                        self.progress_var.set(50 + (i / len(vertices)) * 30)
                        self.root.update()

                self.progress_var.set(80)
                self.status_var.set("Writing triangle...")
                self.root.update()

                # Write Triangle
                for i, face in enumerate(faces):
                    f.write(f"TRGL {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
                    if i % 1000 == 0:
                        self.progress_var.set(80 + (i / len(faces)) * 15)
                        self.root.update()

                if len(vertices) > 0:
                    f.write(f"BSTONE {1}\n")


                if len(vertices) > 1:
                    f.write(f"BORDER {1} {1} {1}\n")

                f.write("END\n")

            self.progress_var.set(100)
            self.status_var.set(f"Conversion complete. Output saved to: {output_file}")
            messagebox.showinfo("Success", f"The STL file has been successfully converted to GOCAD TSurf format. Output file: {output_file}")

        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"An error occurred during conversion:\n{str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = StlToGocadConverter(root)
    root.mainloop()
