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
                               confusion_matrix, save_path=None):
    """
    绘制四个模型的训练指标对比图，包含loss、lr、accuracy和混淆矩阵

    参数:
    - dataset1_path (str): 第一个模型数据路径
    - dataset2_path (str): 第二个模型数据路径
    - dataset3_path (str): 第三个模型数据路径
    - dataset4_path (str): 第四个模型数据路径
    - model1_label (str): 第一个模型的legend标签
    - model2_label (str): 第二个模型的legend标签
    - model3_label (str): 第三个模型的legend标签
    - model4_label (str): 第四个模型的legend标签
    - confusion_matrix (ndarray): 混淆矩阵数据
    - save_path (str, optional): 保存图像的路径
    """
    # 创建2×2的画布
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 设置子图间距
    plt.subplots_adjust(hspace=0.4, wspace=0.3, left=0.1, right=0.95, top=0.95, bottom=0.1)

    # 设置绘图颜色 - 四个不同的颜色
    colors = ['#d7191c', '#006300', '#3ecec9', '#ff7f0e']

    # 模型数据路径和标签
    dataset_paths = [dataset1_path, dataset2_path, dataset3_path, dataset4_path]
    model_labels = [model1_label, model2_label, model3_label, model4_label]

    # 子图标签
    subplot_labels = ['(a)', '(b)', '(c)', '(d)']

    # 提取所有模型的数据
    all_loss_data = []
    all_lr_data = []
    all_accuracy_data = []

    for path in dataset_paths:
        file_path = f'{path}/litho_GNN_log.txt'
        loss_df = extract_litho_loss_data(file_path)
        lr_df = extract_lr_data(file_path)
        accuracy_df = extract_accuracy_data(file_path)

        all_loss_data.append(loss_df)
        all_lr_data.append(lr_df)
        all_accuracy_data.append(accuracy_df)

    # 绘制Loss曲线 (左上角)
    ax_loss = axes[0, 0]
    for i, (loss_df, label, color) in enumerate(zip(all_loss_data, model_labels, colors)):
        ax_loss.plot(loss_df['Epoch'], loss_df['Litho Loss'], color=color, marker='o',
                     markersize=3, label=label, linewidth=2)

    ax_loss.set_xlabel('训练周期', fontsize=26, labelpad=10)
    ax_loss.set_ylabel('Litho Loss', fontsize=26)
    ax_loss.tick_params(axis='both', labelsize=24)
    ax_loss.grid(True, linestyle='-', color='gray', linewidth=0.5, alpha=0.7)
    ax_loss.legend(fontsize=20, loc='upper right')

    # 在训练周期正下方添加子图标签
    ax_loss.text(0.5, -0.24, subplot_labels[0], transform=ax_loss.transAxes,
                 fontsize=26, fontweight='bold', va='center', ha='center')

    # 绘制LR曲线 (右上角)
    ax_lr = axes[0, 1]
    for i, (lr_df, label, color) in enumerate(zip(all_lr_data, model_labels, colors)):
        ax_lr.plot(lr_df['Epoch'], lr_df['LR'], color=color, marker='o',
                   markersize=3, label=label, linewidth=2)

    ax_lr.set_xlabel('训练周期', fontsize=26, labelpad=10)
    ax_lr.set_ylabel('学习率', fontsize=26)
    ax_lr.tick_params(axis='both', labelsize=24)
    ax_lr.grid(True, linestyle='-', color='gray', linewidth=0.5, alpha=0.7)
    ax_lr.legend(fontsize=20, loc='upper right')

    # 设置Y轴为科学计数法
    ax_lr.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax_lr.get_yaxis().get_major_formatter().set_powerlimits((-3, 3))

    # 在训练周期正下方添加子图标签
    ax_lr.text(0.5, -0.24, subplot_labels[1], transform=ax_lr.transAxes,
               fontsize=26, fontweight='bold', va='center', ha='center')

    # 绘制Accuracy曲线 (左下角)
    ax_acc = axes[1, 0]
    for i, (accuracy_df, label, color) in enumerate(zip(all_accuracy_data, model_labels, colors)):
        ax_acc.plot(accuracy_df['Epoch'], accuracy_df['Accuracy'], color=color, marker='o',
                    markersize=3, label=label, linewidth=2)

    ax_acc.set_xlabel('训练周期', fontsize=26, labelpad=10)
    ax_acc.set_ylabel('准确率', fontsize=26)
    ax_acc.tick_params(axis='both', labelsize=24)
    ax_acc.grid(True, linestyle='-', color='gray', linewidth=0.5, alpha=0.7)
    ax_acc.legend(fontsize=20, loc='lower right')

    # 在训练周期正下方添加子图标签
    ax_acc.text(0.5, -0.24, subplot_labels[2], transform=ax_acc.transAxes,
                fontsize=26, fontweight='bold', va='center', ha='center')

    # 绘制混淆矩阵 (右下角)
    ax_conf = axes[1, 1]

    # 用于混淆矩阵的自定义色带
    custom_cmap = ['#FFFBD5', '#d9eedf', '#C0E2CA', '#9DD3AF']

    # 计算每行的总和，计算百分比
    row_sums = np.sum(confusion_matrix, axis=1, keepdims=True)
    percentage_matrix = confusion_matrix / row_sums * 100

    # 定义颜色边界
    cmap = mcolors.ListedColormap(custom_cmap)
    bounds = [0, 25, 50, 75, 100]  # 颜色分界线
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # 绘制热力图
    sns.heatmap(
        percentage_matrix,
        annot=False,
        fmt='.1f',
        cmap=cmap,
        norm=norm,
        xticklabels=['1', '2', '3', '4'],
        yticklabels=['1', '2', '3', '4'],
        cbar=False,
        linewidths=1,
        linecolor='black',
        square=True,
        ax=ax_conf
    )

    # 在每个单元格中显示分类个数和百分比
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            count = confusion_matrix[i, j]
            percentage = percentage_matrix[i, j]
            text = f'{count}\n({percentage:.1f}%)'
            ax_conf.text(j + 0.5, i + 0.5, text, ha='center', va='center', color='black', fontsize=14)

    # 设置混淆矩阵标签
    ax_conf.set_xlabel('预测岩性', fontsize=26, labelpad=10)
    ax_conf.set_ylabel('真实岩性', fontsize=26)
    ax_conf.tick_params(axis='both', which='major', labelsize=22)

    # 添加色带
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax_conf)
    cbar_ax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(ax_conf.get_children()[0], cax=cbar_ax, ticks=bounds)
    cbar.set_label('百分比', fontsize=26)
    cbar.ax.tick_params(labelsize=24)

    # 在色带中绘制黑色分界线
    for bound in bounds:
        cbar_ax.axhline(bound, color='black', linewidth=0.5)

    # 在预测岩性正下方添加子图标签
    ax_conf.text(0.5, -0.24, subplot_labels[3], transform=ax_conf.transAxes,
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
    model1_label = "M1:GAT+Fault"
    model2_label = "M2:GraphSAGE+Fault"
    model3_label = "M3:GAT+GraphSAGE"  # 请替换为实际模型名称
    model4_label = "M4:GAT+GraphSAGE+Fault"  # 请替换为实际模型名称

    # 设置混淆矩阵数据（可以选择任意一个模型的混淆矩阵）
    confusion_matrix = np.array([
        [72, 0, 0, 0],
        [0, 72, 0, 0],
        [0, 0, 87, 0],
        [0, 0, 0, 84]
    ])

    # 保存路径
    save_path = '../tetra_output_files/Fault_cut/VS/Litho_model_comparison_plots.png'

    # 绘制四模型对比图
    plot_four_model_comparison(dataset1_path, dataset2_path, dataset3_path, dataset4_path,
                               model1_label, model2_label, model3_label, model4_label,
                               confusion_matrix, save_path)