import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib.cm as cm
# 设置中文字体为宋体
plt.rcParams['font.serif'] = ['SimSun']

# 设置英文字体为 Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.sans-serif'] = ['Times New Roman']

# dataset_name = "tetra_output_files/0407_GZ_HRBF_point_25m/0408GAT+SAGE_Epoch10"
dataset_name = "Fault_cut/0407cut_fault2m_regular&random_result/断层编码效果分析/0407regular_GAT+GraphSAGE_nocut_fault_epoch_300128layer"
cross_section_xz ='cross_section_xz'
cross_section_yz ='cross_section_yz'
file_path = f'../tetra_output_files/{dataset_name}/rock_training_results_rock.csv'
save_path_xz = f'../tetra_output_files/{dataset_name}/{cross_section_xz}'
save_path_yz = f'../tetra_output_files/{dataset_name}/{cross_section_yz}'
df = pd.read_csv(file_path)

# 查看数据结构
print(df.head())


def plot_xz_profile(df, save_path=None,space = None):
    # 选择 Y 值最大的位置
    max_y = df['original_coords_y'].max()-80
    df_xz = df[df['original_coords_y'] == max_y]
    # 提取XZ平面的数据
    x = df_xz['original_coords_x']
    z = df_xz['original_coords_z']
    predicted_rock_units = df_xz['predicted_rock_units']
    predicted_level = df_xz['predicted_level']

    # 创建一个二维网格用于绘图, 设置较低的分辨率避免网格化过于明显
    grid_x, grid_z = np.meshgrid(np.arange(min(x), max(x), space), np.arange(min(z), max(z), space))

    # 使用线性插值来填充网格数据
    predicted_rock_units_grid = griddata((x, z), predicted_rock_units, (grid_x, grid_z), method='nearest')
    predicted_level_grid = griddata((x, z), predicted_level, (grid_x, grid_z), method='linear')

    # 绘制predicted_rock_units的XZ剖面图
    # 从 'viridis' 中提取四个离散颜色
    n_colors = 4
    viridis_discrete = cm.get_cmap('viridis', n_colors)
    colors = [viridis_discrete(i) for i in range(n_colors)]
    cmap = ListedColormap(colors)  # 创建离散颜色映射
    bounds = np.array([0.5, 1.5, 2.5, 3.5, 4.5])  # 定义颜色边界
    norm = BoundaryNorm(bounds, cmap.N)

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.imshow(predicted_rock_units_grid, extent=(min(x), max(x), min(z), max(z)),
               origin='lower', aspect='auto', cmap=cmap, norm=norm,
               interpolation='nearest')  # interpolation='nearest' 禁用插值
    xz_rock_cbar = plt.colorbar(label='标量场值')  # 绘制颜色条
    xz_rock_cbar.set_label('标量场值', size=24)  # 设置颜色条标题
    xz_rock_cbar.set_ticks(np.arange(1, 5))  # 设置颜色条刻度为 1-4
    plt.xlabel('X方向', size=24)
    plt.ylabel('Z方向', size=24)
    # plt.title('岩性单元XZ剖面图', size=16)
    plt.tick_params(axis='both', labelsize=24)
    xz_rock_cbar.ax.tick_params(labelsize=24)
    plt.tight_layout()
    # 如果提供了保存路径，则保存图像
    if save_path:
        plt.savefig(f'{save_path}_rock_units.png', dpi=600,transparent=True)
    plt.show()

    # 绘制predicted_level的XZ剖面图
    plt.figure(figsize=(10, 6))
    plt.imshow(predicted_level_grid, extent=(min(x), max(x), min(z), max(z)), origin='lower', aspect='auto',
               cmap='viridis')
    xz_level_cbar=plt.colorbar(label='标量场值 ',ticks=np.linspace(0,  80, 5))
    xz_level_cbar.set_label('标量场值', size=24)
    plt.xlabel('X方向',size=24)
    plt.ylabel('Z方向',size=24)
    # plt.title('标量场XZ剖面图',size=16)
    plt.tick_params(axis='both', labelsize=24)
    xz_level_cbar.ax.tick_params(labelsize=24)
    # 如果提供了保存路径，则保存图像
    if save_path:
        plt.savefig(f'{save_path}_level.png',dpi=600,transparent=True)
    else:
        plt.show()


# 创建YZ剖面图
def plot_yz_profile(df, save_path=None,space = None):
    # 选择 X 值最大的位置
    max_x = df['original_coords_x'].max()
    df_yz = df[df['original_coords_x'] == max_x]

    # 提取YZ平面的数据
    y = df_yz['original_coords_y']
    z = df_yz['original_coords_z']
    predicted_rock_units = df_yz['predicted_rock_units']
    predicted_level = df_yz['predicted_level']

    # 创建一个二维网格用于绘图, 设置较低的分辨率避免网格化过于明显
    grid_y, grid_z = np.meshgrid(np.arange(min(y), max(y), space), np.arange(min(z), max(z),space))

    # 使用线性插值来填充网格数据
    predicted_rock_units_grid = griddata((y, z), predicted_rock_units, (grid_y, grid_z), method='nearest')
    predicted_level_grid = griddata((y, z), predicted_level, (grid_y, grid_z), method='linear')

    # 绘制predicted_rock_units的YZ剖面图
    n_colors = 4
    viridis_discrete = cm.get_cmap('viridis', n_colors)
    colors = [viridis_discrete(i) for i in range(n_colors)]
    cmap = ListedColormap(colors)  # 创建离散颜色映射
    bounds = np.array([0.5, 1.5, 2.5, 3.5, 4.5])  # 定义颜色边界
    norm = BoundaryNorm(bounds, cmap.N)

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.imshow(predicted_rock_units_grid, extent=(min(y), max(y), min(z), max(z)),
               origin='lower', aspect='auto', cmap=cmap, norm=norm,
               interpolation='nearest')  # interpolation='nearest' 禁用插值
    xz_rock_cbar = plt.colorbar(label='标量场值')  # 绘制颜色条
    xz_rock_cbar.set_label('标量场值', size=24)  # 设置颜色条标题
    xz_rock_cbar.set_ticks(np.arange(1, 5))  # 设置颜色条刻度为 1-4
    plt.tick_params(axis='both', labelsize=24)
    xz_rock_cbar.ax.tick_params(labelsize=24)
    plt.xlabel('Y方向',size=24)
    plt.ylabel('Z方向', size=24)
    # plt.title('岩性单元YZ剖面图', size=16)

    plt.tight_layout()
    # 如果提供了保存路径，则保存图像
    if save_path:
        plt.savefig(f'{save_path}_rock_units.png', dpi=600, transparent=True)

    plt.show()

    # 绘制predicted_level的YZ剖面图
    plt.figure(figsize=(10, 6))
    plt.imshow(predicted_level_grid, extent=(min(y), max(y), min(z), max(z)), origin='lower', aspect='auto',
               cmap='viridis')
    yz_level_cbar = plt.colorbar(label='标量场值 ',ticks=np.linspace(0, 100, 6))
    yz_level_cbar.set_label('标量场值', size=24)
    plt.xlabel('Y方向',size=24)
    plt.ylabel('Z方向',size=24)
    # plt.title('标量场YZ剖面图',size=16)
    plt.tick_params(axis='both', labelsize=24)
    yz_level_cbar.ax.tick_params(labelsize=24)
    # 如果提供了保存路径，则保存图像
    if save_path:
        plt.savefig(f'{save_path}_level.png', dpi=600, transparent=True)

    plt.show()

if __name__ == '__main__':
    plot_xz_profile(df, save_path=save_path_xz,space=1)
    plot_yz_profile(df, save_path=save_path_yz,space=1)


