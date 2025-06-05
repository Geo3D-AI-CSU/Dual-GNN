# 提取带有"Fault_occurence"字符串的节点
import numpy as np
def extract_fault_occurence_nodes(graph_data):
    # 确保属性为字符串类型
    attributes = graph_data.attribute.astype(str)
    # 检查是否以 "Fault" 开头并以 "occurence" 结尾
    starts_with_fault = np.char.startswith(attributes, 'Fault')
    ends_with_occurence = np.char.endswith(attributes, 'occurence')
    fault_mask = starts_with_fault & ends_with_occurence
    fault_nodes = np.where(fault_mask)[0]
    return fault_nodes

# 提取带有"conformity"的节点
def extract_conformity_nodes(graph_data):
    """
    提取属性为 'Conformity' 的节点
    """
    conformity_nodes = []
    for idx, attrs in enumerate(graph_data.attribute):
        if isinstance(attrs, str) and 'Conformity' in attrs:
            conformity_nodes.append(idx)
        elif isinstance(attrs, (list, tuple)) and 'Conformity' in attrs:
            conformity_nodes.append(idx)
    return conformity_nodes


# 提取带有"Rock_unit"的节点
def extract_rock_nodes(graph_data):
    """
    提取属性为 'Rock_unit' 的节点
    """
    Rock_nodes = []
    for idx, attrs in enumerate(graph_data.attribute):
        if isinstance(attrs, str) and 'Rock_unit' in attrs:
            Rock_nodes .append(idx)
        elif isinstance(attrs, (list, tuple)) and 'Rock_unit' in attrs:
            Rock_nodes .append(idx)
    return Rock_nodes


# 提取带有"Horizon_occurence"字符串的节点
def extract_horizon_nodes(graph_data):
    attributes = graph_data.attribute.astype(str)
    # 检查是否以 "Horizon" 开头并以 "occurence" 结尾
    starts_with_horizon = np.char.startswith(attributes, 'Horizon')
    ends_with_occurence = np.char.endswith(attributes, 'occurence')
    horizon_mask = starts_with_horizon & ends_with_occurence
    horizon_nodes = np.where(horizon_mask)[0]
    return horizon_nodes

# 提取属性完全为"Fault"的节点
def extract_fault_nodes(graph_data):
    attributes = graph_data.attribute.astype(str)
    # 检查是否完全等于 "Fault"
    exact_fault_mask = attributes == 'Fault'
    fault_nodes = np.where(exact_fault_mask)[0]
    return fault_nodes
