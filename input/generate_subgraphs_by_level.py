import numpy as np
from torch_geometric.data import Data
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
# Split subgraphs by Level value
# Group by Level value and extract the coordinates and gradient values for each group.
def split_subgraphs_by_level(graph_data, fault_nodes):
    levels = graph_data.level[fault_nodes]
    unique_levels = np.unique(levels)

    fault_groups = {}
    for level in unique_levels:
        level_nodes = fault_nodes[levels == level]
        coords = graph_data.original_coords[level_nodes]
        gradients = graph_data.gradient[level_nodes]

        # Extract edges associated with nodes
        mask_edge = np.isin(graph_data.edge_index[0], level_nodes) | np.isin(graph_data.edge_index[1], level_nodes)
        sub_edge_index = graph_data.edge_index[:, mask_edge]

        fault_groups[level] = {
            'nodes': level_nodes,
            'coords': coords,
            'gradients': gradients,
            'edge_index': sub_edge_index
        }
    return fault_groups

def create_subgraph(graph_data, nodes, edge_index):
    sub_x = graph_data.x[nodes]

    # Convert the node list into a set to accelerate lookup operations.
    node_set = set(nodes.tolist())

    # Filter out edges that exist only within the subgraph
    mask_edge = [ (src in node_set) and (dst in node_set) for src, dst in edge_index.t().tolist() ]
    filtered_edge_index = edge_index[:, mask_edge]

    # Reindexing nodes: Creating a mapping from the global index to the subgraph's local index
    node_map = {node: idx for idx, node in enumerate(nodes.tolist())}
    # Reindex edges using mapping
    src = filtered_edge_index[0].tolist()
    dst = filtered_edge_index[1].tolist()
    local_src = [node_map[src_node] for src_node in src]
    local_dst = [node_map[dst_node] for dst_node in dst]
    local_edge_index = torch.tensor([local_src, local_dst], dtype=torch.long).to(device)

    # Extract labels and masks
    sub_level = graph_data.level[nodes].clone().detach().float().to(device)
    sub_mask_level = torch.zeros_like(graph_data.level, dtype=torch.bool).to(device)
    sub_mask_level[nodes] = True

    sub_rock_unit = graph_data.rock_unit[nodes].clone().detach().long().to(device)
    sub_mask_rock_unit = torch.zeros_like(graph_data.rock_unit, dtype=torch.bool).to(device)
    sub_mask_rock_unit[nodes] = True

    # Processing raw coordinates and gradients
    if isinstance(graph_data.original_coords, torch.Tensor):
        original_coords = graph_data.original_coords[nodes].clone().detach().float().to(device)
    else:
        original_coords = torch.tensor(graph_data.original_coords[nodes], dtype=torch.float32).clone().detach().to(device)

    if isinstance(graph_data.gradient, torch.Tensor):
        gradient = graph_data.gradient[nodes].clone().detach().float().to(device)
    else:
        gradient = torch.tensor(graph_data.gradient[nodes], dtype=torch.float32).clone().detach().to(device)

    # Data objects for creating subgraphs
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

    # Add nodes property
    subgraph.nodes = nodes

    return subgraph
