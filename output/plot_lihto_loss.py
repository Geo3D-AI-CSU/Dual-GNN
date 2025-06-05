import matplotlib.pyplot as plt
import pandas as pd
import re  # 用于提取周期
import matplotlib as mpl
# 设置中文字体为黑体
# 设置中文字体为宋体
plt.rcParams['font.family'] = 'SimSun'
plt.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题


def plot_rock_loss(file_path, save_path=None):
    """
    绘制训练过程中Rock Unit Loss的曲线图，并保存为图片文件。

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
            # 解析每行数据，使用正则表达式提取周期数字和rock unit loss
            match = re.match(r"Epoch (\d+)/\d+:  Litho Loss = ([\d.]+)", line.strip())
            if match:
                # 提取周期和损失值
                epoch = int(match.group(1))  # 提取周期
                rock_loss = float(match.group(2))  # 提取 rock unit loss

                # 将数据添加到列表
                data.append([epoch, rock_loss])

    # 将数据转换为Pandas DataFrame
    df = pd.DataFrame(data, columns=['Epoch', 'Rock Unit Loss'])

    # 创建图像
    plt.figure(figsize=(13, 8))
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)  # 调整图框范围
    # 绘制 Rock Unit Loss 曲线
    plt.plot(df['Epoch'], df['Rock Unit Loss'], label='Litho Loss', color='#d7191c', marker='o', markersize=6)

    # 设置标题和标签
    plt.xlabel('训练周期', fontsize=28)
    plt.ylabel('Litho Loss', fontsize=28)
    plt.legend(loc='upper right',fontsize=24)
    plt.tick_params(axis='y', labelsize=28)  # 设置左侧Y轴刻度数字大小
    plt.tick_params(axis='x', labelsize=28)  # 设置左侧Y轴刻度数字大小
    plt.grid(True, which='both', axis='both', linestyle='-', color='gray', linewidth=0.7, alpha=0.7)
    # 显示图像
    plt.tight_layout()

    # 如果提供了保存路径，则保存图像
    if save_path:
        plt.savefig(save_path, dpi=600)
        print(f"图像已保存至 {save_path}")

    plt.show()
if __name__ == '__main__':
    dataset_name = "Fault_cut/0407cut_fault2m_regular&random_result/嵌入模块性能分析/0407regularGraphSAGE_cut_fault_epoch_300128layer"
    file_path = f'../tetra_output_files/{dataset_name}/litho_GNN_log.txt'
    save_path=f'../tetra_output_files/{dataset_name}/litho_loss.png'
    plot_rock_loss(file_path , save_path)