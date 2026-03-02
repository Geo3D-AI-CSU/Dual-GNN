import torch
import numpy as np
if not hasattr(np,'bool'):
    np.bool = bool
import pyvista as pv
from input.select_device import select_device
import platform
import os
from scipy import ndimage

device = select_device(desired_gpu=0)


def create_fault_surface1(level, fault_df, global_bounds, visualize=False, output_dir='fault_surfaces', grid_resolution=200, gaussian_sigma=1.0):
    """
    Creates a 3D fault surface based on the given level and fault data.
    Uses PyVista's contour method for surface reconstruction.

    Parameters:
    - level (int): Fault level.
    - fault_df (pd.DataFrame): Fault data containing 'Level', 'X', 'Y', 'Z' columns.
    - global_bounds (tuple): Global boundaries (x_min, x_max, y_min, y_max, z_min, z_max).
    - visualize (bool): Whether to visualize the fault surface.
    - output_dir (str): Directory to save visualization images.
    - grid_resolution (int): Voxel grid resolution.
    - gaussian_sigma (float): Sigma for Gaussian smoothing.

    Returns:
    - surface (pv.PolyData): Generated fault surface.
    """
    # Extract points for the current level
    points = fault_df[fault_df['Level'] == level][['X', 'Y', 'Z']].values

    if len(points) < 4:
        raise ValueError(f"Level {level} ")

    # Create a PyVista point cloud
    point_cloud = pv.PolyData(points)

    # Define global bounds
    x_min, x_max, y_min, y_max, z_min, z_max = global_bounds

    # Create voxel grid using ImageData
    grid = pv.ImageData()
    grid.origin = (x_min, y_min, z_min)
    grid.spacing = (
        (x_max - x_min) / grid_resolution,
        (y_max - y_min) / grid_resolution,
        (z_max - z_min) / grid_resolution
    )
    grid.dimensions = (grid_resolution, grid_resolution, grid_resolution)

    # Initialize scalar field
    scalar_field = np.zeros(grid.dimensions, dtype=np.float32)

    # Map fault points to voxel grid and assign level value
    for point in points:
        ix = int((point[0] - x_min) / grid.spacing[0])
        iy = int((point[1] - y_min) / grid.spacing[1])
        iz = int((point[2] - z_min) / grid.spacing[2])
        if 0 <= ix < grid_resolution and 0 <= iy < grid_resolution and 0 <= iz < grid_resolution:
            scalar_field[ix, iy, iz] = level  # Assign level value

    # Apply Gaussian smoothing
    scalar_field = ndimage.gaussian_filter(scalar_field, sigma=gaussian_sigma)

    # Assign scalar field to grid
    grid.point_data["LevelScalar"] = scalar_field.flatten(order="F")

    # Extract isosurface
    contour_value = level-0.5   # Adjust threshold as needed
    surface = grid.contour(isosurfaces=[contour_value], scalars="LevelScalar")

    # Visualization
    if visualize:
        try:
            current_system = platform.system()
            if current_system in ["Windows", "Darwin"]:  # Supports GUI
                # Interactive window
                surface.plot(show_edges=True, color='lightblue', show_normals=False)
            else:
                # Save image on headless server
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                screenshot_path = os.path.join(output_dir, f"fault_surface_level_{level}.png")
                plotter = pv.Plotter(off_screen=True)
                plotter.add_mesh(surface, color='lightblue', show_edges=True)
                plotter.camera_position = 'xy'  # Fixed camera position
                plotter.show(auto_close=False)
                plotter.screenshot(screenshot_path)
                plotter.close()
        except Exception as e:
            print(f" Level {level} ")
    # Ensure normal vectors are consistent (pointing towards upper plate)
    surface.compute_normals(inplace=True, consistent_normals=True, split_vertices=True)

    return surface

def create_bounding_box1(coords, buffer=1.0):
    """
    Create a bounding box encompassing all nodes.

    Parameters:
    - coords (numpy.ndarray): Node coordinates (M, 3).
    - buffer (float): The buffer size added to each side of the bounding box.

    Returns:
    - bounding_box (tuple): Bounding boxes encompassing all nodes, formatted as(x_min, x_max, y_min, y_max, z_min, z_max)
    """
    x_min, y_min, z_min = coords.min(axis=0) - buffer
    x_max, y_max, z_max = coords.max(axis=0) + buffer
    bounding_box = (x_min, x_max, y_min, y_max, z_min, z_max)
    return bounding_box

    return graph_data
