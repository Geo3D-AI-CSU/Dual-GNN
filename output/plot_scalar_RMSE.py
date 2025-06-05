import matplotlib.pyplot as plt
import pandas as pd
import re  # 用于提取周期
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
from holoviews.plotting.bokeh.styles import font_size

# 设置中文字体为黑体
plt.rcParams['font.family'] = 'SimSun'
plt.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题


def plot_rmse(file_path, save_path=None):
    """
    绘制训练过程中 RMSE 的曲线图，并保存为图片文件。

    参数:
    - file_path (str): 训练日志文件的路径，包含周期、RMSE 的值。
    - save_path (str, optional): 如果提供该路径，则将图像保存到指定路径。如果为None，图像只会显示而不保存。

    返回:
    - None
    """
    # 读取txt文件中的数据，并将其转换为DataFrame
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 使用正则表达式提取 Epoch 和 RMSE 的值
            match = re.match(
                r"Epoch (\d+)/\d+:.*?RMSE = ([\d.]+)",
                line.strip()
            )
            if match:
                epoch = int(match.group(1))
                rmse = float(match.group(2))
                data.append([epoch, rmse])

        # 转换为 DataFrame
    df = pd.DataFrame(data, columns=['Epoch', 'RMSE'])

    # 创建图像，包含 RMSE 曲线
    plt.figure(figsize=(15, 8))
    # 调整图框与图像边缘的间距
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)  # 调整图框范围
    plt.plot(df['Epoch'], df['RMSE'], label='RMSE', color='#5959fb', marker='o', markersize=6)
    plt.xlabel('训练周期', fontsize=40)
    plt.ylabel('RMSE', fontsize=40)
    plt.legend(loc='upper right',fontsize=34)
    plt.tick_params(axis='y', labelsize=40)  # 设置左侧Y轴刻度数字大小
    plt.tick_params(axis='x', labelsize=40)  # 设置左侧Y轴刻度数字大小
    plt.grid(True, which='both', axis='both', linestyle='-', color='gray', linewidth=0.7, alpha=0.7)

    # 如果提供了保存路径，则保存图像
    if save_path:
        plt.savefig(save_path, dpi=600)
        print(f"图像已保存至 {save_path}")

    plt.show()


if __name__ == '__main__':
    dataset_name = "Fault_cut/0407cut_fault2m_regular&random_result/嵌入模块性能分析/0407regularGATcut_fault300epoch_128layer"
    file_path = f'../tetra_output_files/{dataset_name}/scalar_GNN_log.txt'
    save_path=f'../tetra_output_files/{dataset_name}/scalar_rmse.png'
    plot_rmse(file_path , save_path)  # 如果不想保存图片可以将 save_path 设置为 None
