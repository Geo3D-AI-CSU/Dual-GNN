import numpy as np

# 读取文件内容
dataset_name = "Fault_cut/0407cut_fault2m_regular&random_result/嵌入模块性能分析/0407regularGATcut_fault300epoch_128layer"
file_path = f"../tetra_output_files/{dataset_name}/litho_GNN_log.txt"  # 替换为你的文件路径



with open(file_path, "r") as f:
    lines = f.readlines()

# 找到混淆矩阵的开始和结束位置
confusion_matrix_start = None
confusion_matrix_end = None
for i, line in enumerate(lines):
    if "Confusion Matrix:" in line:
        confusion_matrix_start = i + 1
    if "Litho Training Time:" in line:
        confusion_matrix_end = i
        break

# 提取混淆矩阵部分
if confusion_matrix_start is not None and confusion_matrix_end is not None:
    confusion_matrix_lines = lines[confusion_matrix_start:confusion_matrix_end]
    # 将每行转换为列表
    confusion_matrix = []
    for line in confusion_matrix_lines:
        row = line.strip().replace("[", "").replace("]", "").split()
        row = [int(x) for x in row]
        confusion_matrix.append(row)
    confusion_matrix = np.array(confusion_matrix)
else:
    raise ValueError("未找到混淆矩阵！")

# 打印提取的混淆矩阵
print("提取的混淆矩阵：")
print(confusion_matrix)

# 如果需要格式化混淆矩阵（例如添加逗号）
def format_matrix(matrix):
    formatted_matrix = []
    for row in matrix:
        formatted_row = "[" + ", ".join(map(str, row)) + "]"
        formatted_matrix.append(formatted_row)
    formatted_matrix = "[\n" + ",\n".join(formatted_matrix) + "\n]"
    return formatted_matrix

# 格式化混淆矩阵
formatted_confusion_matrix = format_matrix(confusion_matrix)
print("\n格式化后的混淆矩阵：")
print(formatted_confusion_matrix)