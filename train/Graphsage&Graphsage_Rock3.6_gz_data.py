import torch
import time
import os
import json
from torch.nn.functional import dropout

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from input.input_graph_data import create_or_load_graph
from model.model import GATLevelPredictor, RockUnitPredictor
# GATRockPredictor  、
from loss.loss_fn import level_loss, gradient_loss,rock_unit_loss
from utils.metrics import calculate_rmse, calculate_accuracy, calculate_r2, calculate_confusion_matrix
from input.extract_data_from_attribute import  extract_horizon_nodes
from input.select_device import select_device, set_random_seed
from input.extract_data_from_attribute import extract_conformity_nodes,extract_rock_nodes
from output.save_data import save_rock_result_to_csv
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from loss.GradNorm import GradNorm_2loss
from input.Normalizer import  Normalizer
from input.compute_fault_zone_feature import compute_fault_features


set_random_seed(42)
device = select_device(desired_gpu=0)
normalizer = Normalizer()

def train_and_validate(graph_data,min_values, max_values,num_epochs=300,lr=0.01,hidden_channels=128,num_classes=7,result_dir=None,dropout=dropout,lr_decay=0.8):
    # Initialise the model
    model =  GATLevelPredictor(graph_data.x.size(1),hidden_channels=hidden_channels,dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.01)

    # Using the ReduceLROnPlateau scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min', 
        factor=lr_decay,  
        patience=10,  
        threshold=1e-2,  
        min_lr=1e-6  
    )

    mask_level =graph_data.mask_level
    mask_rock_unit = graph_data.mask_rock_unit
    mask_gradient = graph_data.mask_gradient
    original_level =graph_data.level[mask_level]
    level_norm = normalizer.fit_transform_level_masked(graph_data.level[mask_level])
    graph_data.level[mask_level] = level_norm
    edge_index = graph_data.edge_index.to(device)
    gradient = graph_data.gradient.to(device)
    original_coords = graph_data.original_coords.to(device)
    start_time = time.time()

    # Initialise GradNorm
    grad_norm = GradNorm_2loss(alpha=1.0, gamma=0, device=device)

    log_scalar_file = os.path.join(result_dir, 'scalar_GNN_log.txt')
    with open(log_scalar_file, 'w') as f:
        for epoch in range(1, num_epochs + 1):
            model.train()
            optimizer.zero_grad()
            # Prediction Level
            predicted_level = model(graph_data.x.to(device), graph_data.edge_index.to(device))

            # Calculate Level loss (only for masked nodes)
            level_loss_val = level_loss(predicted_level[mask_level], graph_data.level[mask_level].to(device))
            if mask_gradient.any():
                grad_loss_val = gradient_loss(
                    predicted_level,
                    original_coords,
                    gradient[mask_gradient, 0],
                    gradient[mask_gradient, 1],
                    gradient[mask_gradient, 2],
                    edge_index,
                    mask_gradient
                )
            # Update GradNorm weights
            loss_weights = grad_norm.update_weights(level_loss_val, gradient_loss=grad_loss_val,model=model)
            # Calculate the total loss
            total_loss = grad_norm.compute_loss(level_loss_val, grad_loss_val)

            # Perform backpropagation and update the model
            total_loss.backward()
            optimizer.step()
            # Update learning rate (based on total loss)
            scheduler.step(total_loss)  
            if epoch % 10 == 0:
                # Retrieve the current learning rate
                current_lr = optimizer.param_groups[0]['lr']  
                rmse = calculate_rmse(predicted_level, graph_data.level, mask_level)
                r2 = calculate_r2(predicted_level, graph_data.level, mask_level)
                print(
                    f"Epoch {epoch}/{num_epochs}: Interface Loss = {level_loss_val.item():.4f}, "
                    f"Orientation Loss = {grad_loss_val.item():.4f}, "
                    f" Total Loss = {total_loss.item():.4f}, RMSE = {rmse:.4f}, "
                    f"R2 = {r2:.4f}, LR = {current_lr:.6f}"
                )
                f.write(
                    f"Epoch {epoch}/{num_epochs}: Interface Loss = {level_loss_val.item():.4f}, "
                    f"Orientation Loss = {grad_loss_val.item():.4f}, "
                    f"Total Loss = {total_loss.item():.4f}, RMSE = {rmse:.4f},"
                    f"R2 = {r2:.4f}, LR = {current_lr:.6f}\n"
                )
        end_time = time.time()
        level_time = end_time - start_time

    with open(log_scalar_file, 'a') as f:
        f.write(f"Scalar GNN Training Time: {level_time:.2f} seconds\n")
        model.eval()
        with torch.no_grad():
            predicted_level = model(graph_data.x.to(device), graph_data.edge_index.to(device))  
            predicted_level_original = normalizer.inverse_transform_level(predicted_level)
        # rock_unit_model = GATRockPredictor(graph_data.x.size(1)+1, hidden_channels=hidden_channels, num_classes=num_classes,dropout=dropout).to(device)
        rock_unit_model = RockUnitPredictor(graph_data.x.size(1), hidden_channels=128, num_classes=num_classes).to(device)
        rock_unit_optimizer = torch.optim.AdamW(rock_unit_model.parameters(), lr=lr,weight_decay=0.01)
        scheduler_rock = ReduceLROnPlateau(
            rock_unit_optimizer,
            mode='min',  
            factor=0.5,  
            patience=10,  
            threshold=1e-2,  
            min_lr=1e-6  
        )
    log_litho_file = os.path.join(result_dir, 'litho_GNN_log.txt')
    with open(log_litho_file, 'w') as f: 
        start_time_rock = time.time()
        for epoch in range(1, num_epochs + 1):
            rock_unit_model.train()
            rock_unit_optimizer.zero_grad()
            # Convert the predicted level to a two-dimensional tensor of shape (num_nodes, 1)
            predicted_level_expanded = predicted_level.view(-1, 1)
            # Use the predicted level and the x feature from the original image as input.
            rock_unit_input = torch.cat([predicted_level_expanded, graph_data.x.to(device)], dim=-1)
            predicted_rock_units = rock_unit_model(rock_unit_input, graph_data.edge_index.to(device))
            mask_rock_unit = graph_data.mask_rock_unit.to(device)
            rock_unit_loss_value = rock_unit_loss(predicted_rock_units[mask_rock_unit], graph_data.rock_unit.to(device)[mask_rock_unit])
            # Backpropagate and update parameters
            rock_unit_loss_value.backward()
            rock_unit_optimizer.step()
            # Update learning rate
            scheduler_rock.step(rock_unit_loss_value) 
            if epoch % 10 == 0:
                accuracy = calculate_accuracy(predicted_rock_units, graph_data.rock_unit, mask_rock_unit)
                current_lr_rock = rock_unit_optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}/{num_epochs}: Litho Loss = {rock_unit_loss_value.item():.4f}, Accuracy = {accuracy:.4f}, LR = {current_lr_rock:.6f}")
                f.write(
                f"Epoch {epoch}/{num_epochs}:  Litho Loss = {rock_unit_loss_value.item():.4f}, Accuracy = {accuracy:.4f},LR = {current_lr_rock:.6f}\n",
            )
        confusion_matrix_result = calculate_confusion_matrix(predicted_rock_units, graph_data.rock_unit, mask_rock_unit)
        f.write(f"Confusion Matrix:\n{confusion_matrix_result}\n")
        end_time_rock = time.time()
        rock_time = end_time_rock - start_time_rock

    with open(log_litho_file, 'a') as f:  
        f.write(f"Litho Training Time: {rock_time:.2f} seconds\n")
    rock_unit_model.eval()
    with torch.no_grad():
        rock_unit_input = torch.cat([predicted_level_expanded , graph_data.x.to(device)], dim=-1)
        predicted_rock_units = rock_unit_model(rock_unit_input, graph_data.edge_index.to(device))
        predicted_rock_units = torch.argmax(predicted_rock_units, dim=-1).cpu().numpy()  
        all_nodes = np.arange(graph_data.x.size(0))
        fault_features = graph_data.x[:, 3:].cpu().numpy()
    save_rock_result_to_csv(
            graph_data=graph_data,
            predicted_level= predicted_level_original,
            fault_features=fault_features,
            nodes=all_nodes,
            predicted_rock_units=predicted_rock_units+1,
            suffix='_rock',
            result_dir=result_dir
    )
    return model

def main(node_file, ele_file,vtk_file,epoch=300,
         lr=0.01,hidden_channels=128,
         num_classes=6,min_values=[-9999,250,500,750],
         max_values=[250,500,750,-9999],result_dir=None,
         factor=1.0,dropout=0,is_gradient = False,lr_decay=0.8):

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    graph_data = create_or_load_graph(node_file, ele_file,is_gradient=is_gradient)
    # graph_data = compute_fault_features(graph_data,vtk_file,factor=factor)
    # Training and validation, passing levels
    trained_model = train_and_validate(graph_data,num_epochs=epoch,lr=lr,hidden_channels=hidden_channels,
                                       num_classes=num_classes,min_values=min_values, max_values=max_values,
                                       result_dir=result_dir,dropout=dropout,lr_decay=lr_decay)
    # Save the trained model
    torch.save(trained_model.state_dict(), os.path.join(result_dir, 'model.pth'))
    # Record training parameters and log information
    log_file = os.path.join(result_dir, 'model_parameter_log.txt')
    with open(log_file, 'w') as f:
        f.write("Model Parameter Log\n")
        f.write(f"Epochs: {epoch}\n")
        f.write(f"Learning Rate: {lr}\n")
        f.write(f"Learning Rate decay: {lr_decay}\n")
        f.write(f"Hidden Channels: {hidden_channels}\n")
        f.write(f"Number of Classes: {num_classes}\n")
        f.write(f"Min Values: {min_values}\n")
        f.write(f"Max Values: {max_values}\n")
        f.write(f"Fault amplification: {factor}\n")
        f.write("Training completed successfully.\n")

def run_experiment(params):
    main(
        node_file=params["node_file"],
        ele_file=params["ele_file"],
        vtk_file=params["vtk_file"],
        epoch=params["epoch"],
        lr=params["lr"],
        hidden_channels=params["hidden_channels"],
        num_classes=params["num_classes"],
        min_values=params["min_values"],
        max_values=params["max_values"],
        factor=params["factor"],
        dropout=params["dropout"],
        is_gradient=params["is_gradient"],
        lr_decay=params["lr_decay"],
        result_dir = params["result_dir"]
    )


if __name__ == "__main__":
    #  Read the experimental parameter configuration file
    with open('Training parameter settings/0226GAT_gz_data.json', 'r') as f:
        experiments = json.load(f)
    # Iterate through each set of experimental parameters and run
    for experiment_params in experiments:
        experiment_params[
            "result_dir"] = (f"../tetra_output_files/0226_GZ_HRBF_point/GraphSAGE+SAGE_Epoch{experiment_params['epoch']}_0.1litho_noFault")
            # "result_dir"] = (f"../tetra_output_files/2.21plate/0221GAT_Epoch{experiment_params['epoch']}"
            # f"_NoFault_LRdecay{experiment_params['lr_decay']}/")
        run_experiment(experiment_params)
        print('************************************************************************************************************\n'
              f'Model_Epoch{experiment_params["epoch"]}_Fault{experiment_params["factor"]}_Training completed successfully.\n'
              '************************************************************************************************************\n\n\n')
