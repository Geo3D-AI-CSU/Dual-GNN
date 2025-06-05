import matplotlib.pyplot as plt
import pandas as pd
import re
import matplotlib.ticker as ticker
import matplotlib as mpl
import numpy as np
import seaborn as sns
import matplotlib.colors as mcolors

# 设置中文字体
plt.rcParams['font.family'] = 'SimSun'
plt.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def extract_litho_loss_data(file_path):
    """提取岩性损失数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.match(r"Epoch (\d+)/\d+:  Litho Loss = ([\d.]+)", line.strip())
            if match:
                epoch = int(match.group(1))
                rock_loss = float(match.group(2))
                data.append([epoch, rock_loss])
    return pd.DataFrame(data, columns=['Epoch', 'Litho Loss'])


def extract_lr_data(file_path):
    """提取学习率数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.match(
                r"Epoch (\d+)/\d+:  Litho Loss = [\d.]+, Accuracy = [\d.]+,LR = ([\d.]+)",
                line.strip())
            if match:
                epoch = int(match.group(1))
                lr = float(match.group(2))
                data.append([epoch, lr])
    return pd.DataFrame(data, columns=['Epoch', 'LR'])


def extract_accuracy_data(file_path):
    """提取准确率数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.match(
                r"^Epoch (\d+)/\d+:\s*Litho\s*Loss\s*=\s*[\d.]+,\s*Accuracy\s*=\s*([\d.]+),\s*LR\s*=\s*[\d.]+",
                line.strip())
            if match:
                epoch = int(match.group(1))
                accuracy = float(match.group(2))
                data.append([epoch, accuracy])
    return pd.DataFrame(data, columns=['Epoch', 'Accuracy'])


def plot_four_model_comparison(dataset1_path, dataset2_path, dataset3_path, dataset4_path,
                               model1_label, model2_label, model3_label, model4_label,
                               save_path=None):
    """
    绘制四个模型的训练指标对比图，包含loss和accuracy

    参数:
    - dataset1_path (str): 第一个模型数据路径
    - dataset2_path (str): 第二个模型数据路径
    - dataset3_path (str): 第三个模型数据路径
    - dataset4_path (str): 第四个模型数据路径
    - model1_label (str): 第一个模型的legend标签
    - model2_label (str): 第二个模型的legend标签
    - model3_label (str): 第三个模型的legend标签
    - model4_label (str): 第四个模型的legend标签
    - save_path (str, optional): 保存图像的路径
    """
    # 创建1×2的画布
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    # 设置子图间距
    plt.subplots_adjust(hspace=0.4, wspace=0.3, left=0.1, right=0.95, top=0.95, bottom=0.1)

    # 设置绘图颜色 - 四个不同的颜色
    colors = ['#2b83ba', '#fdae61', '#d7191c', '#006300']

    # 模型数据路径和标签
    dataset_paths = [dataset1_path, dataset2_path, dataset3_path, dataset4_path]
    model_labels = [model1_label, model2_label, model3_label, model4_label]

    # 子图标签
    subplot_labels = ['(a)', '(b)']

    # 提取所有模型的数据
    all_loss_data = []
    all_accuracy_data = []

    for path in dataset_paths:
        file_path = f'{path}/litho_GNN_log.txt'
        loss_df = extract_litho_loss_data(file_path)
        accuracy_df = extract_accuracy_data(file_path)

        all_loss_data.append(loss_df)
        all_accuracy_data.append(accuracy_df)

    # 绘制Loss曲线 (左图)
    ax_loss = axes[0]
    for i, (loss_df, label, color) in enumerate(zip(all_loss_data, model_labels, colors)):
        ax_loss.plot(loss_df['Epoch'], loss_df['Litho Loss'], color=color, marker='o',
                     markersize=3, label=label, linewidth=2)

    ax_loss.set_xlabel('训练周期', fontsize=26, labelpad=10)
    ax_loss.set_ylabel('Litho Loss', fontsize=26)
    ax_loss.tick_params(axis='both', labelsize=24)
    ax_loss.grid(True, linestyle='-', color='gray', linewidth=0.5, alpha=0.7)
    ax_loss.legend(fontsize=20, loc='upper right')

    # 在训练周期正下方添加子图标签
    ax_loss.text(0.5, -0.25, subplot_labels[0], transform=ax_loss.transAxes,
                 fontsize=26, fontweight='bold', va='center', ha='center')

    # 绘制Accuracy曲线 (右图)
    ax_acc = axes[1]
    for i, (accuracy_df, label, color) in enumerate(zip(all_accuracy_data, model_labels, colors)):
        ax_acc.plot(accuracy_df['Epoch'], accuracy_df['Accuracy'], color=color, marker='o',
                    markersize=3, label=label, linewidth=2)

    ax_acc.set_xlabel('训练周期', fontsize=26, labelpad=10)
    ax_acc.set_ylabel('准确率', fontsize=26)
    ax_acc.tick_params(axis='both', labelsize=24)
    ax_acc.grid(True, linestyle='-', color='gray', linewidth=0.5, alpha=0.7)
    ax_acc.legend(fontsize=20, loc='lower right')

    # 在训练周期正下方添加子图标签
    ax_acc.text(0.5, -0.25, subplot_labels[1], transform=ax_acc.transAxes,
                fontsize=26, fontweight='bold', va='center', ha='center')

    # 调整布局
    plt.tight_layout()

    # 如果提供了保存路径，则保存图像
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"图像已保存至 {save_path}")

    plt.show()


if __name__ == '__main__':
    # 设置四个模型的数据路径
    dataset1_path = '../tetra_output_files/Fault_cut/VS/embedding/0407regularGATcut_fault300epoch_128layer/'
    dataset2_path = '../tetra_output_files/Fault_cut/VS/embedding/0407regularGraphSAGE_cut_fault_epoch_300128layer/'
    dataset3_path = '../tetra_output_files/Fault_cut/VS/fautl/0407regular_GAT+GraphSAGE_nocut_fault_epoch_300128layer/'
    dataset4_path = '../tetra_output_files/Fault_cut/VS/fautl/0407GATregular+GraphSAGE_cut_fault_epoch_300128layer/'

    # 设置四个模型的legend标签
    model1_label = "M1:FRep-DualGAT"
    model2_label = "M2:FRep-DualGraphSAGE"
    model3_label = "M3:GAT-GraphSAGE"  # 请替换为实际模型名称
    model4_label = "M4:FRep-GAT-GraphSAGE"  # 请替换为实际模型名称

    # 保存路径
    save_path = '../tetra_output_files/Fault_cut/VS/Litho_model_comparison_plots.png'

    # 绘制四模型对比图
    plot_four_model_comparison(dataset1_path, dataset2_path, dataset3_path, dataset4_path,
                               model1_label, model2_label, model3_label, model4_label, save_path)