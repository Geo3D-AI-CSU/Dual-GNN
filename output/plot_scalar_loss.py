import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import re  # 用于提取周期
import matplotlib.font_manager as fm
import matplotlib
import matplotlib as mpl

plt.rcParams['font.family'] = 'SimSun'
plt.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

def plot_training_loss(file_path, save_path=None):
    """
    绘制训练过程中Level Loss、Gradient Loss、Total Loss和RMSE的曲线图，并保存为图片文件。

    参数:
    - file_path (str): 训练日志文件的路径，包含周期和损失值。
    - save_path (str, optional): 如果提供该路径，则将图像保存到指定路径。如果为None，图像只会显示而不保存。

    返回:
    - None
    """
    # 读取txt文件中的数据，并将其转换为DataFrame
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 解析每行数据，使用正则表达式提取周期数字
            match = re.match(
                r"Epoch\s*(\d+)/\d+:\s*Interface Loss\s*=\s*([\d.]+),\s*Orientation Loss\s*=\s*([\d.]+),\s*Total Loss\s*=\s*([\d.]+),\s*RMSE\s*=\s*([\d.]+),\s*R2\s*=\s*([\d.]+),\s*LR\s*=\s*([\d.]+)",
                line.strip())

            if match:
                # 提取周期和损失值
                epoch = int(match.group(1))  # 提取周期
                level_loss = float(match.group(2))
                gradient_loss = float(match.group(3))
                total_loss = float(match.group(4))
                rmse = float(match.group(5))

                # 将数据添加到列表
                data.append([epoch, level_loss, gradient_loss, total_loss, rmse])

    # 将数据转换为Pandas DataFrame
    df = pd.DataFrame(data, columns=['Epoch', 'Level Loss', 'Gradient Loss', 'Total Loss', 'RMSE'])

    # 创建一个图像，包含 Level Loss、Gradient Loss、Total Loss 和 RMSE
    fig, ax1 = plt.subplots(figsize=(15, 8))
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)  # 调整图框范围
    # 绘制 Level Loss, Gradient Loss 和 Total Loss 曲线
    ax1.plot(df['Epoch'], df['Level Loss'], label='Interface Loss', color='#2b83ba', marker='s', markersize=6)
    ax1.plot(df['Epoch'], df['Gradient Loss'], label='Orientation Loss', color='#fdae61', marker='^', markersize=6)
    ax1.plot(df['Epoch'], df['Total Loss'], label='Total Loss', color='#d7191c', marker='*', markersize=6)

    # 设置左侧 Y 轴标签和标题
    ax1.set_xlabel('训练周期',fontsize=40)
    ax1.set_ylabel('损失值',fontsize=40)

    # 设置右侧图例（Loss 和 RMSE 都放在右边）
    ax1.legend(loc='upper right')
    # 设置左侧Y轴刻度数字大小
    ax1.tick_params(axis='y', labelsize=40)  # 左侧Y轴刻度数字大小
    # 设置X轴刻度数字大小
    ax1.tick_params(axis='x', labelsize=40)  # 水平X轴刻度数字大小
    # 添加虚线网格
    ax1.grid(True, which='both', axis='both', linestyle='-', color='gray', linewidth=0.7, alpha=0.7)

    # 显示网格
    ax1.set_axisbelow(True)  # 使网格线在图像下方

    # 设置右上角图例显示所有曲线
    ax1.legend(loc='upper right', markerscale=1,fontsize=34)
    # 显示图像
    plt.tight_layout()

    # 如果提供了保存路径，则保存图像
    if save_path:
        plt.savefig(save_path, dpi=600)
        print(f"图像已保存至 {save_path}")

    plt.show()

# 调用函数绘制 RMSE 曲线
if __name__ == '__main__':

    dataset_name = "Fault_cut/0407cut_fault2m_regular&random_result/嵌入模块性能分析/0407regularGATcut_fault300epoch_128layer"
    file_path = f'../tetra_output_files/{dataset_name}/scalar_GNN_log.txt'
    save_path=f'../tetra_output_files/{dataset_name}/scalar_loss.png'
    plot_training_loss(file_path , save_path)  # 如果不想保存图片可以将 save_path 设置为 None