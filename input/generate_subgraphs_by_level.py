import numpy as np
from torch_geometric.data import Data
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
# 按Level值分割子图
# 按Level值分组并提取每组的坐标和梯度值
def split_subgraphs_by_level(graph_data, fault_nodes):
    levels = graph_data.level[fault_nodes]
    unique_levels = np.unique(levels)

    fault_groups = {}
    for level in unique_levels:
        level_nodes = fault_nodes[levels == level]
        coords = graph_data.original_coords[level_nodes]
        gradients = graph_data.gradient[level_nodes]

        # 提取与这些节点相关的边
        mask_edge = np.isin(graph_data.edge_index[0], level_nodes) | np.isin(graph_data.edge_index[1], level_nodes)
        sub_edge_index = graph_data.edge_index[:, mask_edge]

        fault_groups[level] = {
            'nodes': level_nodes,
            'coords': coords,
            'gradients': gradients,
            'edge_index': sub_edge_index
            # 移除 'edge_attr' 相关内容
        }
    return fault_groups



import torch
from torch_geometric.data import Data
import numpy as np

def create_subgraph(graph_data, nodes, edge_index):
    sub_x = graph_data.x[nodes]

    # 将节点列表转换为集合以加速查找
    node_set = set(nodes.tolist())

    # 筛选出仅在子图内的边
    mask_edge = [ (src in node_set) and (dst in node_set) for src, dst in edge_index.t().tolist() ]
    filtered_edge_index = edge_index[:, mask_edge]

    # 重新索引节点：创建一个从全局索引到子图本地索引的映射
    node_map = {node: idx for idx, node in enumerate(nodes.tolist())}
    # 使用映射重新索引边
    src = filtered_edge_index[0].tolist()
    dst = filtered_edge_index[1].tolist()
    local_src = [node_map[src_node] for src_node in src]
    local_dst = [node_map[dst_node] for dst_node in dst]
    local_edge_index = torch.tensor([local_src, local_dst], dtype=torch.long).to(device)

    # 提取标签和掩码
    sub_level = graph_data.level[nodes].clone().detach().float().to(device)
    sub_mask_level = torch.zeros_like(graph_data.level, dtype=torch.bool).to(device)
    sub_mask_level[nodes] = True

    sub_rock_unit = graph_data.rock_unit[nodes].clone().detach().long().to(device)
    sub_mask_rock_unit = torch.zeros_like(graph_data.rock_unit, dtype=torch.bool).to(device)
    sub_mask_rock_unit[nodes] = True

    # 处理原始坐标和梯度
    if isinstance(graph_data.original_coords, torch.Tensor):
        original_coords = graph_data.original_coords[nodes].clone().detach().float().to(device)
    else:
        original_coords = torch.tensor(graph_data.original_coords[nodes], dtype=torch.float32).clone().detach().to(device)

    if isinstance(graph_data.gradient, torch.Tensor):
        gradient = graph_data.gradient[nodes].clone().detach().float().to(device)
    else:
        gradient = torch.tensor(graph_data.gradient[nodes], dtype=torch.float32).clone().detach().to(device)

    # 创建子图的数据对象
    subgraph = Data(
        x=sub_x.clone().detach().float().to(device),
        edge_index=local_edge_index,
        level=sub_level,
        mask_level=sub_mask_level.clone().detach().bool().to(device),
        rock_unit=sub_rock_unit,
        mask_rock_unit=sub_mask_rock_unit.clone().detach().bool().to(device),
        original_coords=original_coords,
        gradient=gradient
    )

    # 添加 nodes 属性
    subgraph.nodes = nodes

    return subgraph
