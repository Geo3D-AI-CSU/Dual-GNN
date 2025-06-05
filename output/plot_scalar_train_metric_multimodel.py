import matplotlib.pyplot as plt
import pandas as pd
import re
import matplotlib.ticker as ticker
import matplotlib as mpl
import numpy as np

# 设置中文字体
plt.rcParams['font.family'] = 'SimSun'
plt.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def extract_training_data(file_path):
    """
    从训练日志中提取所有需要的数据

    参数:
    - file_path (str): 训练日志文件的路径

    返回:
    - pd.DataFrame: 包含所有提取数据的DataFrame
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 使用正则表达式提取所有需要的数据
            match = re.match(
                r"Epoch\s*(\d+)/\d+:\s*Interface Loss\s*=\s*([\d.]+),\s*Orientation Loss\s*=\s*([\d.]+),\s*Total Loss\s*=\s*([\d.]+),\s*RMSE\s*=\s*([\d.]+),\s*R2\s*=\s*([\d.-]+),\s*LR\s*=\s*([\d.]+)",
                line.strip())

            if match:
                epoch = int(match.group(1))
                level_loss = float(match.group(2))
                gradient_loss = float(match.group(3))
                total_loss = float(match.group(4))
                rmse = float(match.group(5))
                r2 = float(match.group(6))
                lr = float(match.group(7))

                data.append([epoch, level_loss, gradient_loss, total_loss, rmse, r2, lr])

    # 将数据转换为DataFrame
    df = pd.DataFrame(data, columns=['Epoch', 'Interface Loss', 'Orientation Loss', 'Total Loss', 'RMSE', 'R2', 'LR'])
    return df


def plot_four_models_comparison(dataset_paths, model_names, save_path=None):
    """
    绘制2×2的子图，分别显示4个模型的Loss、LR、RMSE和R2曲线

    参数:
    - dataset_paths (list): 包含4个数据集路径的列表
    - model_names (list): 包含4个模型名称的列表，用于图例
    - save_path (str, optional): 图像保存路径

    返回:
    - None
    """
    # 定义颜色和标记列表，确保每个模型有独特的样式
    colors = ['#2b83ba', '#fdae61', '#d7191c', '#006300']
    markers = ['o', 's', '^', '*']

    # 从日志文件中提取数据
    dfs = []
    for path in dataset_paths:
        dfs.append(extract_training_data(path))

    # 创建2×2子图布局
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # 子图标签
    labels = ['(a)', '(b)', '(c)', '(d)']

    # 1. 损失函数曲线 (左上角)
    for i, (df, color, marker, name) in enumerate(zip(dfs, colors, markers, model_names)):
        axes[0, 0].plot(df['Epoch'], df['Total Loss'], label=f'{name}', color=color, marker=marker, markersize=4)

    axes[0, 0].set_xlabel('训练周期', fontsize=26)
    axes[0, 0].set_ylabel('损失值', fontsize=26)
    axes[0, 0].legend(loc='upper right', fontsize=20)
    axes[0, 0].grid(True, linestyle='-', color='gray', linewidth=0.5, alpha=0.5)
    axes[0, 0].tick_params(axis='both', labelsize=24)
    # 修改：将标签放在x轴标签（训练周期）的正下方
    axes[0, 0].text(0.5, -0.2, labels[0], transform=axes[0, 0].transAxes,
                    fontsize=26, fontweight='bold', ha='center')

    # 2. 学习率变化曲线 (右上角)
    for i, (df, color, marker, name) in enumerate(zip(dfs, colors, markers, model_names)):
        axes[0, 1].plot(df['Epoch'], df['LR'], label=f'{name}', color=color, marker=marker, markersize=4)

    axes[0, 1].set_xlabel('训练周期', fontsize=26)
    axes[0, 1].set_ylabel('学习率', fontsize=26)
    axes[0, 1].legend(loc='lower left', fontsize=20)
    axes[0, 1].grid(True, linestyle='-', color='gray', linewidth=0.5, alpha=0.5)
    axes[0, 1].tick_params(axis='both', labelsize=24)
    # 修改：将标签放在x轴标签（训练周期）的正下方
    axes[0, 1].text(0.5, -0.2, labels[1], transform=axes[0, 1].transAxes,
                    fontsize=26, fontweight='bold', ha='center')
    # 设置Y轴科学计数法
    axes[0, 1].yaxis.set_major_formatter(ticker.ScalarFormatter())
    axes[0, 1].yaxis.get_major_formatter().set_powerlimits((-3, 3))

    # 3. 均方根误差变化曲线 (左下角)
    for i, (df, color, marker, name) in enumerate(zip(dfs, colors, markers, model_names)):
        axes[1, 0].plot(df['Epoch'], df['RMSE'], label=f'{name}', color=color, marker=marker, markersize=4)

    axes[1, 0].set_xlabel('训练周期', fontsize=26)
    axes[1, 0].set_ylabel('RMSE', fontsize=26)
    axes[1, 0].legend(loc='upper right', fontsize=20)
    axes[1, 0].grid(True, linestyle='-', color='gray', linewidth=0.5, alpha=0.5)
    axes[1, 0].tick_params(axis='both', labelsize=24)
    # 修改：将标签放在x轴标签（训练周期）的正下方
    axes[1, 0].text(0.5, -0.2, labels[2], transform=axes[1, 0].transAxes,
                    fontsize=26, fontweight='bold', ha='center')

    # 4. 决定系数变化曲线 (右下角)
    for i, (df, color, marker, name) in enumerate(zip(dfs, colors, markers, model_names)):
        axes[1, 1].plot(df['Epoch'], df['R2'], label=f'{name}', color=color, marker=marker, markersize=4)

    axes[1, 1].set_xlabel('训练周期', fontsize=26)
    axes[1, 1].set_ylabel('R²', fontsize=26)
    axes[1, 1].legend(loc='lower right', fontsize=20)
    axes[1, 1].grid(True, linestyle='-', color='gray', linewidth=0.5, alpha=0.5)
    axes[1, 1].tick_params(axis='both', labelsize=24)
    # 修改：将标签放在x轴标签（训练周期）的正下方
    axes[1, 1].text(0.5, -0.2, labels[3], transform=axes[1, 1].transAxes,
                    fontsize=26, fontweight='bold', ha='center')

    # 调整子图间距，为下方标签留出空间
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)  # 增加这行来为底部标签留出更多空间

    # 如果提供了保存路径，保存图像
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"图像已保存至 {save_path}")

    plt.show()


if __name__ == '__main__':
    # 在此处设置四个数据集的路径
    dataset1_path = '../tetra_output_files/Fault_cut/VS/embedding/0407regularGATcut_fault300epoch_128layer/scalar_GNN_log.txt'
    dataset2_path = '../tetra_output_files/Fault_cut/VS/embedding/0407regularGraphSAGE_cut_fault_epoch_300128layer/scalar_GNN_log.txt'
    dataset3_path = '../tetra_output_files/Fault_cut/VS/fautl/0407regular_GAT+GraphSAGE_nocut_fault_epoch_300128layer/scalar_GNN_log.txt'
    dataset4_path = '../tetra_output_files/Fault_cut/VS/fautl/0407GATregular+GraphSAGE_cut_fault_epoch_300128layer/scalar_GNN_log.txt'

    # 在此处设置模型名称（用于图例）
    model1_name = 'M1:FRep-DualGAT'
    model2_name = 'M2:FRep-DualGraphSAGE'
    model3_name = 'M3:GAT-GraphSAGE'
    model4_name = 'M4:FRep-GAT-GraphSAGE'

    # 设置图像保存路径
    save_path = '../tetra_output_files/Fault_cut/VS/Scalar_model_comparison_plots.png'

    # 整合数据集路径和模型名称
    dataset_paths = [dataset1_path, dataset2_path, dataset3_path, dataset4_path]
    model_names = [model1_name, model2_name, model3_name, model4_name]

    # 绘制对比图
    plot_four_models_comparison(dataset_paths, model_names, save_path)