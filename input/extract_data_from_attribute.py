# Extract nodes containing the string 'Fault_occurrence'
import numpy as np
def extract_fault_occurence_nodes(graph_data):
    # Ensure the attribute is of string type
    attributes = graph_data.attribute.astype(str)
    # Check whether it begins with 'Fault' and ends with 'occurrence'.
    starts_with_fault = np.char.startswith(attributes, 'Fault')
    ends_with_occurence = np.char.endswith(attributes, 'occurence')
    fault_mask = starts_with_fault & ends_with_occurence
    fault_nodes = np.where(fault_mask)[0]
    return fault_nodes

# Extract nodes containing 'conformity'
def extract_conformity_nodes(graph_data):
    conformity_nodes = []
    for idx, attrs in enumerate(graph_data.attribute):
        if isinstance(attrs, str) and 'Conformity' in attrs:
            conformity_nodes.append(idx)
        elif isinstance(attrs, (list, tuple)) and 'Conformity' in attrs:
            conformity_nodes.append(idx)
    return conformity_nodes


# Extract nodes containing 'Rock_unit'
def extract_rock_nodes(graph_data):
    Rock_nodes = []
    for idx, attrs in enumerate(graph_data.attribute):
        if isinstance(attrs, str) and 'Rock_unit' in attrs:
            Rock_nodes .append(idx)
        elif isinstance(attrs, (list, tuple)) and 'Rock_unit' in attrs:
            Rock_nodes .append(idx)
    return Rock_nodes


# Extract nodes containing the string 'Horizon_occurrence'
def extract_horizon_nodes(graph_data):
    attributes = graph_data.attribute.astype(str)
    # Check whether it begins with 'Horizon' and ends with 'occurence'.
    starts_with_horizon = np.char.startswith(attributes, 'Horizon')
    ends_with_occurence = np.char.endswith(attributes, 'occurence')
    horizon_mask = starts_with_horizon & ends_with_occurence
    horizon_nodes = np.where(horizon_mask)[0]
    return horizon_nodes

# Extract nodes whose attributes are entirely of the 'Fault' type
def extract_fault_nodes(graph_data):
    attributes = graph_data.attribute.astype(str)
    # Check whether it is exactly equal to 'Fault'
    exact_fault_mask = attributes == 'Fault'
    fault_nodes = np.where(exact_fault_mask)[0]
    return fault_nodes

