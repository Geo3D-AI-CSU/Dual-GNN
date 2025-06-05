import pandas as pd
import re  # 用于提取周期
import matplotlib.font_manager as fm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl

# 设置中文字体为黑体
plt.rcParams['font.family'] = 'SimSun'
plt.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题


def plot_lr(file_path, save_path=None):
    """
    绘制训练过程中 Learning Rate (LR) 的曲线图，并保存为图片文件。

    参数:
    - file_path (str): 训练日志文件的路径，包含周期和LR的值。
    - save_path (str, optional): 如果提供该路径，则将图像保存到指定路径。如果为None，图像只会显示而不保存。

    返回:
    - None
    """
    # 读取txt文件中的数据，并将其转换为DataFrame
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 解析每行数据，使用正则表达式提取周期数字和学习率（LR）的值
            match = re.match(
                r"Epoch (\d+)/\d+:  Litho Loss = [\d.]+, Accuracy = [\d.]+,LR = ([\d.]+)",
                line.strip())
            if match:
                # 提取周期和学习率（LR）
                epoch = int(match.group(1))  # 提取周期
                lr = float(match.group(2))

                # 将数据添加到列表
                data.append([epoch, lr])

    # 将数据转换为Pandas DataFrame
    df = pd.DataFrame(data, columns=['Epoch', 'LR'])

    # 创建图像，包含 LR 曲线
    plt.figure(figsize=(13, 8))
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)  # 调整图框范围
    plt.plot(df['Epoch'], df['LR'], label='Learning Rate (LR)', color='#006300', marker='o', markersize=6)
    plt.xlabel('训练周期', fontsize=28)
    plt.ylabel('学习率', fontsize=28)
    plt.legend(loc='upper right',fontsize=24)
    plt.tick_params(axis='y', labelsize=28)  # 设置左侧Y轴刻度数字大小
    plt.tick_params(axis='x', labelsize=28)  # 设置左侧Y轴刻度数字大小

    # 设置Y轴刻度使用科学计数法
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.tick_params(axis='y', which='major', labelsize=28)
    ax.get_yaxis().get_major_formatter().set_powerlimits((-3, 3))  # 设置科学计数法显示范围

    # 添加虚线网格
    plt.grid(True, which='both', axis='both', linestyle='-', color='gray', linewidth=0.7, alpha=0.7)

    # 如果提供了保存路径，则保存图像
    if save_path:
        plt.savefig(save_path, dpi=600)
        print(f"图像已保存至 {save_path}")

    plt.show()

# # 文件路径
if __name__ == '__main__':
    dataset_name = "Fault_cut/0407cut_fault2m_regular&random_result/嵌入模块性能分析/0407regularGraphSAGE_cut_fault_epoch_300128layer"
    file_path = f'../tetra_output_files/{dataset_name}/litho_GNN_log.txt'

    save_path=f'../tetra_output_files/{dataset_name}/litho_lr.png'
    # 调用函数绘制学习率曲线
    plot_lr(file_path, save_path)  # 如果不想保存图片可以将 save_path 设置为 None