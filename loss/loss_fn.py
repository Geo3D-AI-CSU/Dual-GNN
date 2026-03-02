import torch
import torch.nn.functional as F

# Scalar inequality loss function
def scalar_loss(predicted_scalar, rock_unit_labels, min_values, max_values):
    """
    Compute the loss function for the lithological scalar field, 
    ensuring predicted scalar values remain within the specified range, 
    whilst handling missing minimum and maximum values.
    """
    eps = 1e-8  # Preventing zero-division errors
    # Obtain the minimum and maximum values corresponding to the lithological labels
    min_values_for_labels = min_values[rock_unit_labels-1]  # Obtain the minimum value corresponding to each lithological label
    max_values_for_labels = max_values[rock_unit_labels-1]  # Obtain the maximum value corresponding to each lithological label

    # Ensure that min_values and max_values align with predicted_scalar on the device.
    min_values_for_labels = min_values_for_labels.to(predicted_scalar.device)
    max_values_for_labels = max_values_for_labels.to(predicted_scalar.device)

    # Initialise loss value
    loss = torch.zeros_like(predicted_scalar, requires_grad=True)

    # Calculate the loss of scalar values
    min_mask = min_values_for_labels != -9999
    max_mask = max_values_for_labels != -9999

    # Handling cases where min_value is -9999
    loss = torch.where(~min_mask & max_mask & (predicted_scalar > max_values_for_labels),
                        torch.abs(predicted_scalar - max_values_for_labels)/(max_values_for_labels+eps),
                        loss)

    # Handling cases where max_value is -9999
    loss = torch.where(~max_mask & min_mask & (predicted_scalar < min_values_for_labels),
                        torch.abs(predicted_scalar - min_values_for_labels)/(min_values_for_labels+eps),
                        loss)

    # Handling cases where both min_value and max_value are not -9999
    loss = torch.where(min_mask & max_mask & (predicted_scalar > max_values_for_labels),
                        torch.abs(predicted_scalar - max_values_for_labels)/(max_values_for_labels - min_values_for_labels+eps),
                        loss)
    loss = torch.where(min_mask & max_mask & (predicted_scalar < min_values_for_labels),
                        torch.abs(predicted_scalar - min_values_for_labels)/(max_values_for_labels - min_values_for_labels+eps),
                        loss)
    # The sum of the loss values for all nodes
    return loss.mean()*0.1


# SclarGNN scalar value loss function
def level_loss(predicted, target):
    return  F.mse_loss(predicted, target)
# LithoGNN's lithological unit loss function
def rock_unit_loss(predicted_rock_units, rock_unit_labels):
    rock_unit_labels = rock_unit_labels - 1
    return F.cross_entropy(predicted_rock_units, rock_unit_labels)

# The Gradient Loss Function of SclarGNN
def gradient_loss(predicted_levels, coords, dx, dy, dz, edge_index, mask_gradient):
    """
    Compute the gradient loss.
    """
    mask_indices = torch.nonzero(mask_gradient, as_tuple=False).squeeze()
    mask_sample = torch.zeros(mask_indices .size(0), dtype=torch.bool) 
    row, col = edge_index
    gradients = []

    for i, node in enumerate(mask_indices):
        row_mask = (row == node)  # Records whether row equals node
        col_mask = (col == node)  # Record whether col equals node

        # Identify the neighbours within row and col that are associated with the current node.
        col_neighbors = col[row_mask]  # neighbours in the row
        row_neighbors = row[col_mask]  # Neighbours in the column

        # Merge neighbouring rows and columns
        neighbors = torch.cat((row_neighbors, col_neighbors))

        # Update the mask, check for neighbours
        if neighbors.numel() > 0:
            mask_sample[i] = True  
        else:
            mask_sample[i] = False  

        # Skip nodes with no neighbours
        if not mask_sample[i]:
            gradients.append(torch.zeros(3, device=predicted_levels.device))  
            continue  

        neighbors_v = neighbors
        # Calculate delta_coords and delta_levels
        delta_coords = coords[neighbors_v] - coords[node].unsqueeze(0)
        delta_levels = predicted_levels[neighbors_v] - predicted_levels[node].unsqueeze(0)

        # Compute gradients
        delta_coords.requires_grad_(True)
        delta_levels.requires_grad_(True)

        # Compute the transpose of the Jacobian matrix and multiply it by delta_coords
        AtA = torch.matmul(delta_coords.T, delta_coords)
        Atb = torch.matmul(delta_coords.T, delta_levels.unsqueeze(1))

        # Use torch.linalg.pinv for stable solution
        try:
            gradient = torch.linalg.pinv(AtA) @ Atb
            gradients.append(gradient.squeeze())
        except:
            gradients.append(torch.zeros(3, device=predicted_levels.device))
            print(f"Solving fail, setting gradient to [0, 0, 0].")

    # After calculating the gradients for all nodes, stack them into a single tensor.
    gradient_estimates = torch.stack(gradients) if gradients else torch.zeros_like(dx)  

    # True gradient
    true_gradients = torch.stack([dx, dy, dz], dim=-1)

    # Normalised gradient
    norm_predicted_gradients = torch.norm(gradient_estimates, dim=-1, keepdim=True) + 1e-8
    normalized_predicted_gradients = gradient_estimates / norm_predicted_gradients

    norm_true_gradients = torch.norm(true_gradients, dim=-1, keepdim=True) + 1e-8
    normalized_true_gradients = true_gradients / norm_true_gradients
    mask_sample = mask_sample.to(normalized_true_gradients.device)
    # Calculate the gradient loss under the mask
    masked_predicted_gradients = normalized_predicted_gradients * mask_sample.unsqueeze(-1)  
    masked_true_gradients = normalized_true_gradients * mask_sample.unsqueeze(-1) 

    # Calculate the cosine similarity between the predicted gradient and the actual gradient
    cos_theta = (masked_predicted_gradients * masked_true_gradients).sum(dim=-1)
    mask = (cos_theta != 0)
    # Calculate angle loss
    angle_loss = torch.mean(1 - cos_theta[mask])

    return angle_loss*0.1

def gradient_loss_autogradient(predicted_levels, coords, dx, dy, dz, edge_index, mask_gradient):
    """
    Compute the gradient at the coordinates corresponding to the predicted 'level', and compare it with the target gradient (dx, dy, dz) to calculate the angular loss.
    """
    # Positions where mask_gradient is set to True require gradient computation.
    mask_indices = torch.nonzero(mask_gradient, as_tuple=False).squeeze()
    row, col = edge_index  # Obtain the connection information for the edges

    gradients = []

    # Traverse all nodes requiring gradient computation
    for v in mask_indices:
        neighbors_v = col[row == v]
        if neighbors_v.numel() == 0:
            gradients.append(torch.zeros(3, device=predicted_levels.device))
            continue

        # 计算delta_coords和delta_levels
        delta_coords = coords[neighbors_v] - coords[v].unsqueeze(0)
        delta_levels = predicted_levels[neighbors_v] - predicted_levels[v].unsqueeze(0)

        # Gradients computed using automatic differentiation for prediction
        coords_v = coords[v].unsqueeze(0).requires_grad_(True)  
        predicted_levels_v = predicted_levels[v].unsqueeze(0).requires_grad_(True)  

        # Calculate the loss
        loss = torch.abs(predicted_levels_v - predicted_levels[neighbors_v])

        # Clear the gradient to avoid it being affected when calculating other gradients.
        coords_v.grad = None 

        # Backpropagation, calculating gradients
        loss.backward()

        # Obtain the gradient for node v
        gradient = coords_v.grad
        gradients.append(gradient.squeeze())

    # After calculating the gradients for all nodes, stack them into a single tensor.
    gradient_estimates = torch.stack(gradients)

    # True gradient
    true_gradients = torch.stack([dx, dy, dz], dim=-1)

    # Normalised gradient
    norm_predicted_gradients = torch.norm(gradient_estimates, dim=-1, keepdim=True) + 1e-8
    normalized_predicted_gradients = gradient_estimates / norm_predicted_gradients

    norm_true_gradients = torch.norm(true_gradients, dim=-1, keepdim=True) + 1e-8
    normalized_true_gradients = true_gradients / norm_true_gradients

    # Calculate the cosine similarity between the predicted gradient and the actual gradient
    cos_theta = (normalized_predicted_gradients * normalized_true_gradients).sum(dim=-1)

    # Calculate angle loss
    angle_loss = torch.sum(1 - cos_theta)

    return angle_loss

