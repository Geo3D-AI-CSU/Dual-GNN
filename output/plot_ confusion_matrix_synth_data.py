import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import matplotlib
import matplotlib as mpl
# 设置中文字体为黑体
plt.rcParams['font.family'] = 'SimSun'
plt.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

dataset_name = "Fault_cut/0407cut_fault2m_regular&random_result/嵌入模块性能分析/0407regularGATcut_fault300epoch_128layer"
save_path=f'../tetra_output_files/{dataset_name}/custom_confusion_matrix.png'

def plot_confusion_matrix(conf_matrix, save_path=None, custom_cmap=None):
    """
    绘制混淆矩阵热力图，并保存为图片文件，支持自定义色带和在矩阵中显示分类个数和百分比。

    参数:
    - conf_matrix (np.array): 混淆矩阵数据。
    - save_path (str, optional): 如果提供该路径，则将图像保存到指定路径。如果为None，图像只会显示而不保存。
    - custom_cmap (list or str, optional): 用户自定义的色带颜色。如果为None，使用默认色带。

    返回:
    - None
    """
    # 计算每行的总和，计算百分比
    row_sums = np.sum(conf_matrix, axis=1, keepdims=True)
    percentage_matrix = conf_matrix / row_sums * 100

    # 创建图像，绘制热力图
    plt.figure(figsize=(13,8))
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)  # 调整图框范围
    # 自定义色带，如果没有提供则使用默认色带
    if custom_cmap:
        cmap = mcolors.ListedColormap(custom_cmap)
    else:
        cmap = 'Blues'  # 默认蓝色调色带

    # 定义颜色边界
    bounds = [0, 25, 50, 75, 100]  # 颜色分界线
    norm = mcolors.BoundaryNorm(bounds, cmap.N)  # 使用 BoundaryNorm 明确颜色边界

    # 绘制热力图，使用百分比矩阵作为颜色值
    ax = sns.heatmap(
        percentage_matrix,
        annot=False,
        fmt='.1f',
        cmap=cmap,
        norm=norm,  # 使用 BoundaryNorm
        xticklabels=['1', '2', '3', '4'],
        yticklabels=['1', '2', '3', '4'],
        cbar=False,  # 先关闭默认色带
        annot_kws={"size": 24},
        linewidths=1,  # 设置线条宽度
        linecolor='black',  # 设置线条颜色为黑色
        square=True,  # 设置单元格为正方形
    )

    # 在每个单元格中显示分类个数和百分比
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            count = conf_matrix[i, j]
            percentage = percentage_matrix[i, j]
            text = f'{count}\n({percentage:.1f}%)'
            # 控制字体大小和文本位置，避免溢出
            ax.text(j + 0.5, i + 0.5, text, ha='center', va='center', color='black', fontsize=24)

    # 设置标题和标签
    plt.xlabel('预测岩性', fontsize=28)
    plt.ylabel('真实岩性', fontsize=28)

    # 调整x轴和y轴刻度标签的字体大小
    ax.tick_params(axis='both', which='major', labelsize=28)  # 设置x和y轴刻度标签字体大小

    # 手动创建色带
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes("right", size="5%", pad=0.1)  # 创建色带轴
    cbar = plt.colorbar(ax.get_children()[0], cax=cbar_ax, ticks=bounds)  # 创建色带
    cbar.set_label('百分比', fontsize=24)  # 设置色带标签
    cbar.ax.tick_params(labelsize=24)  # 设置色带刻度标签字体大小

    # 在色带中绘制黑色分界线
    for bound in bounds:
        cbar_ax.axhline(bound, color='black', linewidth=0.5)  # 在每个边界处绘制黑色水平线

    # 如果提供了保存路径，则保存图像
    if save_path:
        plt.savefig(save_path, dpi=600)
        print(f"图像已保存至 {save_path}")

    plt.show()


# 手动定义混淆矩阵数据
conf_matrix = np.array([
[72, 0, 0, 0],
[0, 72, 0, 0],
[0, 0, 87, 0],
[0, 0, 0, 84]
])

# 用户自定义色带（例如：由浅绿到深绿的渐变色带）
custom_cmap = ['#FFFBD5','#d9eedf','#C0E2CA', '#9DD3AF']

# 调用函数绘制混淆矩阵并显示/保存
plot_confusion_matrix(conf_matrix, save_path=save_path, custom_cmap=custom_cmap)