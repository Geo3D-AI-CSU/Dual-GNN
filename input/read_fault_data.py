import pandas as pd

def read_fault_data(csv_file):
    """
    读取断层面数据的CSV文件。
    CSV文件应包含列：X, Y, Z, level
    """
    fault_df = pd.read_csv(csv_file)
    # 提取所有唯一的断层名称
    fault_names = fault_df['Level'].unique().tolist()
    return fault_df, fault_names