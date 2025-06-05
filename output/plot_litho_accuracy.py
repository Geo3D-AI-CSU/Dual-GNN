import matplotlib.pyplot as plt
import pandas as pd
import re  # 用于提取准确率（Accuracy）
import matplotlib.font_manager as fm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl

plt.rcParams['font.family'] = 'SimSun'
plt.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

def plot_accuracy(file_path, save_path=None):
    """
    绘制训练过程中 Accuracy 的曲线图，并保存为图片文件。

    参数:
    - file_path (str): 训练日志文件的路径，包含周期和Accuracy的值。
    - save_path (str, optional): 如果提供该路径，则将图像保存到指定路径。如果为None，图像只会显示而不保存。

    返回:
    - None
    """
    # 读取txt文件中的数据，并将其转换为DataFrame
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 解析每行数据，使用正则表达式提取周期数字和Accuracy的值
            match = re.match(
                 r"^Epoch (\d+)/\d+:\s*Litho\s*Loss\s*=\s*[\d.]+,\s*Accuracy\s*=\s*([\d.]+),\s*LR\s*=\s*[\d.]+",
                line.strip())
            if match:
                # 提取周期和Accuracy值
                epoch = int(match.group(1))  # 提取周期
                accuracy = float(match.group(2))   # 提取Accuracy值

                # 将数据添加到列表
                data.append([epoch, accuracy])

    # 将数据转换为Pandas DataFrame
    df = pd.DataFrame(data, columns=['Epoch', 'Accuracy'])

    # 创建图像，包含 Accuracy 曲线
    plt.figure(figsize=(13, 8))
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)  # 调整图框范围
    plt.plot(df['Epoch'], df['Accuracy'], label='Accuracy', color='#3ecec9', marker='o', markersize=6)
    plt.xlabel('训练周期', fontsize=28)
    plt.ylabel('准确率', fontsize=28)
    # plt.legend(loc='upper right')
    plt.tick_params(axis='y', labelsize=28)  # 设置左侧Y轴刻度数字大小
    plt.tick_params(axis='x', labelsize=28)  # 设置X轴刻度数字大小
    plt.grid(True, which='both', axis='both', linestyle='-', color='gray', linewidth=0.7, alpha=0.7)

    # 如果提供了保存路径，则保存图像
    if save_path:
        plt.savefig(save_path, dpi=600)
        print(f"图像已保存至 {save_path}")

    plt.show()


# # 示例使用
if __name__ == '__main__':
    dataset_name = "Fault_cut/0407cut_fault2m_regular&random_result/嵌入模块性能分析/0407regularGATcut_fault300epoch_128layer"
    file_path = f'../tetra_output_files/{dataset_name}/litho_GNN_log.txt'
    save_path=f'../tetra_output_files/{dataset_name}/litho_accuracy.png'
    plot_accuracy(file_path, save_path)  # 如果不想保存图片可以将 save_path 设置为 None
