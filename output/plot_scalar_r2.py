import matplotlib.pyplot as plt
import pandas as pd
import re  # 用于提取R2值
import matplotlib.font_manager as fm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
# 设置中文字体为黑体
# 设置中文字体为黑体
plt.rcParams['font.family'] = 'SimSun'
plt.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题



def plot_r2(file_path, save_path=None):
    """
    绘制训练过程中 R2 值的曲线图，并保存为图片文件。

    参数:
    - file_path (str): 训练日志文件的路径，包含周期和R2的值。
    - save_path (str, optional): 如果提供该路径，则将图像保存到指定路径。如果为None，图像只会显示而不保存。

    返回:
    - None
    """
    # 读取txt文件中的数据，并将其转换为DataFrame
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 使用正则表达式提取 Epoch 和 R2 的值
            match = re.match(
                r"Epoch (\d+)/\d+:\s*Interface Loss = [\d.]+,\s*Orientation Loss = [\d.]+,\s*Total Loss = [\d.]+,\s*RMSE = [\d.]+,\s*R2 = ([\d.-]+),\s*LR = [\d.]+",
                line.strip()
            )
            if match:
                epoch = int(match.group(1))  # 提取 Epoch 值
                r2 = float(match.group(2))  # 提取 R2 值
                data.append([epoch, r2])

    # 转换为 DataFrame
    df = pd.DataFrame(data, columns=['Epoch', 'R2'])

    # 创建图像，包含 R2 曲线
    plt.figure(figsize=(18, 12))
    # 调整图框与图像边缘的间距
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)  # 调整图框范围
    plt.plot(df['Epoch'], df['R2'], label='R2', color='#2b83ba', marker='o', markersize=6)
    plt.xlabel('训练周期', fontsize=40)
    plt.ylabel('R2', fontsize=40)
    # plt.legend(loc='upper right')
    plt.tick_params(axis='y', labelsize=40)  # 设置左侧Y轴刻度数字大小
    plt.tick_params(axis='x', labelsize=40)  # 设置左侧Y轴刻度数字大小
    plt.grid(True, which='both', axis='both', linestyle='-', color='gray', linewidth=0.7, alpha=0.7)
    # 如果提供了保存路径，则保存图像
    if save_path:
        plt.savefig(save_path, dpi=600)
        print(f"图像已保存至 {save_path}")

    plt.show()


if __name__ == '__main__':
    dataset_name = "Fault_cut/0407cut_fault2m_regular&random_result/嵌入模块性能分析/0407regularGraphSAGE_cut_fault_epoch_300128layer"
    file_path = f'../tetra_output_files/{dataset_name}/scalar_GNN_log.txt'
    save_path=f'../tetra_output_files/{dataset_name}/scalar_r2.png'
    plot_r2(file_path , save_path)
