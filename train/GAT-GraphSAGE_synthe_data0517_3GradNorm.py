import torch
import time
import os
import json
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from input.input_graph_data import create_or_load_graph
from model.model import  GATLevelPredictor, GATRockPredictor,RockUnitPredictor
# LevelPredictor
from loss.loss_fn import level_loss, gradient_loss, rock_unit_loss,scalar_loss
from utils.metrics import calculate_rmse, calculate_accuracy, calculate_r2, calculate_confusion_matrix
from input.select_device import select_device, set_random_seed
from output.save_data import save_rock_result_to_csv
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from loss.GradNorm import GradNorm_3loss
from input.Normalizer import  Normalizer
from input.compute_fault_zone_feature import compute_fault_features


set_random_seed(42)
device = select_device(desired_gpu=0)
# 归一化 Level 和 Coordinates
normalizer = Normalizer()

def train_and_validate(graph_data,min_values, max_values,num_epochs=300,lr=0.01,hidden_channels=128,num_classes=7,result_dir=None,lr_decay=0.8):
    # 初始化模型并移动到设备
    model =  GATLevelPredictor(graph_data.x.size(1),hidden_channels=hidden_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.01)

    # 使用 ReduceLROnPlateau 调度器
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',  # 监控指标的模式：'min' 表示损失越小越好
        factor=lr_decay,  # 学习率衰减因子（new_lr = lr * factor）
        patience=10,  # 等待多少个 epoch 指标没有改善后降低学习率
        threshold=1e-2,  # 指标变化的阈值，只有变化超过阈值才认为是改善
        min_lr=1e-6  # 学习率的下限
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

    # 确定断层特征的数量
    start_time = time.time()

    # 初始化 GradNorm
    grad_norm = GradNorm_3loss(alpha=1.0, gamma=1.0, delta=1.0, device=device)
    log_scalar_file = os.path.join(result_dir, 'scalar_GNN_log.txt')
    with open(log_scalar_file, 'w') as f:
        for epoch in range(1, num_epochs + 1):
            model.train()
            optimizer.zero_grad()
            # 预测 Level
            predicted_level = model(graph_data.x.to(device), graph_data.edge_index.to(device))

            # 计算 Level 损失（仅对有掩码的节点）
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
            # 使用 fit_transform_values 来归一化 min_values 和 max_values
            min_values_normalized, max_values_normalized = normalizer.fit_transform_values(min_values, max_values)
            min_values_normalized = torch.tensor(min_values_normalized, requires_grad=True)
            max_values_normalized = torch.tensor(max_values_normalized, requires_grad=True)

            # 计算 scalar loss，确保 predicted_level 在指定范围内
            scalar_loss_val = scalar_loss(predicted_level[mask_rock_unit], graph_data.rock_unit[mask_rock_unit], min_values_normalized,
                                              max_values_normalized)
            # 更新 GradNorm 权重
            loss_weights = grad_norm.update_weights(level_loss_val, gradient_loss=grad_loss_val, scalar_loss=scalar_loss_val,model=model)
            # # 计算总损失
            total_loss = grad_norm.compute_loss(level_loss_val, grad_loss_val,scalar_loss_val)
            # 反向传播并更新模型
            total_loss.backward()
            optimizer.step()
            # 更新学习率（根据总损失）
            scheduler.step(total_loss)  # 使用 total_loss 作为监控指标
            if epoch % 10 == 0:
                # 获取当前学习率
                current_lr = optimizer.param_groups[0]['lr']  # 获取第一个参数组的学习率
                # 计算 RMSE（使用归一化的 level 进行计算）
                rmse = calculate_rmse(predicted_level, graph_data.level, mask_level)
                r2 = calculate_r2(predicted_level, graph_data.level, mask_level)
                print(
                    f"Epoch {epoch}/{num_epochs}: Interface Loss = {level_loss_val.item():.4f}, "
                    f"Orientation Loss = {grad_loss_val.item():.4f},Scalar Loss = {scalar_loss_val.item():.4f}, "
                    f" Total Loss = {total_loss.item():.4f}, RMSE = {rmse:.4f}, "
                    f"R2 = {r2:.4f}, LR = {current_lr:.6f}"
                )
                # 记录到txt文件
                f.write(
                    f"Epoch {epoch}/{num_epochs}: Interface Loss = {level_loss_val.item():.4f}, "
                    f"Orientation Loss = {grad_loss_val.item():.4f}, Scalar Loss = {scalar_loss_val.item():.4f}, "
                    f"Total Loss = {total_loss.item():.4f}, RMSE = {rmse:.4f},"
                    f"R2 = {r2:.4f}, LR = {current_lr:.6f}\n"
                )

        end_time = time.time()
        level_time = end_time - start_time
        print(f"level训练时间: {level_time:.2f} 秒")
            # 将训练时间记录到txt文件

    with open(log_scalar_file, 'a') as f:  # 使用 'a' 模式来追加内容
        f.write(f"Scalar GNN Training Time: {level_time:.2f} seconds\n")
        model.eval()
        with torch.no_grad():
            predicted_level = model(graph_data.x.to(device), graph_data.edge_index.to(device))  # 预测所有节点的 Level
            predicted_level_original = normalizer.inverse_transform_level(predicted_level)
                # 将预测的level与原始图的 x 特征结合

        rock_unit_model = RockUnitPredictor(graph_data.x.size(1), hidden_channels=128, num_classes=num_classes).to(device)
        rock_unit_optimizer = torch.optim.AdamW(rock_unit_model.parameters(), lr=lr,weight_decay=0.01)
        scheduler_rock = ReduceLROnPlateau(
            rock_unit_optimizer,
            mode='min',  # 监控指标的模式：'min' 表示损失越小越好
            factor=0.5,  # 学习率衰减因子（new_lr = lr * factor）
            patience=10,  # 等待多少个 epoch 指标没有改善后降低学习率  # 是否打印学习率更新信息
            threshold=1e-2,  # 指标变化的阈值，只有变化超过阈值才认为是改善
            min_lr=1e-6  # 学习率的下限
        )
    log_litho_file = os.path.join(result_dir, 'litho_GNN_log.txt')
    with open(log_litho_file, 'w') as f: # 使用 'w' 模式来覆盖写入内容
        start_time_rock = time.time()
        for epoch in range(1, num_epochs + 1):
            rock_unit_model.train()
            rock_unit_optimizer.zero_grad()
            # 将预测的 level 转换为 (num_nodes, 1) 形状的二维张量
            predicted_level_expanded = predicted_level.view(-1, 1)
            # 将预测的level与原始图的x特征作为输入
            rock_unit_input = torch.cat([predicted_level_expanded, graph_data.x.to(device)], dim=-1)
            predicted_rock_units = rock_unit_model(rock_unit_input, graph_data.edge_index.to(device))
            mask_rock_unit = graph_data.mask_rock_unit.to(device)
            rock_unit_loss_value = rock_unit_loss(predicted_rock_units[mask_rock_unit], graph_data.rock_unit.to(device)[mask_rock_unit])
            # 反向传播并更新参数
            rock_unit_loss_value.backward()
            rock_unit_optimizer.step()
            # 更新学习率
            scheduler_rock.step(rock_unit_loss_value)  # 使用 rock_unit_loss_value 作为监控指标
                # 记录损失
            if epoch % 10 == 0:
                # 计算准确率
                accuracy = calculate_accuracy(predicted_rock_units, graph_data.rock_unit, mask_rock_unit)

                # 获取当前学习率
                current_lr_rock = rock_unit_optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}/{num_epochs}: Litho Loss = {rock_unit_loss_value.item():.4f}, Accuracy = {accuracy:.4f}, LR = {current_lr_rock:.6f}")
                f.write(
                f"Epoch {epoch}/{num_epochs}:  Litho Loss = {rock_unit_loss_value.item():.4f}, Accuracy = {accuracy:.4f},LR = {current_lr_rock:.6f}\n",
            )
        confusion_matrix_result = calculate_confusion_matrix(predicted_rock_units, graph_data.rock_unit, mask_rock_unit)
        f.write(f"Confusion Matrix:\n{confusion_matrix_result}\n")
        end_time_rock = time.time()
        rock_time = end_time_rock - start_time_rock
        print(f"Litho训练时间: {rock_time:.2f} 秒")
        # 将训练时间记录到txt文件
    with open(log_litho_file, 'a') as f:  # 使用 'a' 模式来追加内容
        f.write(f"Litho Training Time: {rock_time:.2f} seconds\n")
        # 训练结束后保存预测结果
    rock_unit_model.eval()
    with torch.no_grad():
            # 使用训练好的Level预测值和原始特征作为输入进行岩性预测
        rock_unit_input = torch.cat([predicted_level_expanded , graph_data.x.to(device)], dim=-1)
        predicted_rock_units = rock_unit_model(rock_unit_input, graph_data.edge_index.to(device))
        # 获取每个节点预测类别的最大概率
        predicted_rock_units = torch.argmax(predicted_rock_units, dim=-1).cpu().numpy()  # 获取最大概率的类别索引
        # 提取所有节点索引
        all_nodes = np.arange(graph_data.x.size(0))
        # 提取断层特征（假设断层特征位于 graph_data.x 的最后 num_faults 列）
        fault_features = graph_data.x[:, 3:].cpu().numpy()
        # 调用保存函数
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
         num_classes=4,min_values=[-9999,250,500,750],
         max_values=[250,500,750,-9999],result_dir=None,
         factor=1.0,lr_decay=0.8,is_gradient = True):

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    graph_data = create_or_load_graph(node_file, ele_file,is_gradient=is_gradient)
    graph_data = compute_fault_features(graph_data,vtk_file,factor=factor)
    # 训练和验证，传递 levels
    trained_model = train_and_validate(graph_data,num_epochs=epoch,lr=lr,hidden_channels=hidden_channels,
                                       num_classes=num_classes,min_values=min_values, max_values=max_values,result_dir=result_dir,lr_decay=lr_decay)
    # 保存训练好的模型
    torch.save(trained_model.state_dict(), os.path.join(result_dir, 'model.pth'))
    # 记录训练参数及日志信息
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



# 批量训练的逻辑
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
        result_dir=params["result_dir"],
        factor=params["factor"],
        lr_decay=params["lr_decay"],
    )


if __name__ == "__main__":
    # 读取实验参数配置文件
    with open('训练设置参数/0522_syn_3GradNorm_GAT_GraphSAGE.json', 'r') as f:
        experiments = json.load(f)
    # 遍历每组实验参数并运行
    for experiment_params in experiments:
        experiment_params["result_dir"] = (f"../tetra_output_files/Fault_cut/0407cut_fault2m_regular/0522_syn_3GradNorm_GAT_GraphSAGE_{experiment_params['epoch']}"
                             f"{experiment_params['hidden_channels']}layer")
        run_experiment(experiment_params)
        print('************************************************************************************************************\n'
              f'Model_Training completed successfully.\n'
              '************************************************************************************************************\n\n\n')

    # 逐个训练的逻辑
    # node_file = '../tetra_output_files/0214synthetic_data_new/combined_mesh.node'
    # ele_file = '../tetra_output_files/0214synthetic_data_new/combined_mesh.ele'
    # vtk_file = '../tetra_output_files/0214synthetic_data_new/Fault'
    # result_dir = '../train/synthetic_data_new/'
    # epoch=300
    # lr = 0.01
    # hidden_channels = 128
    # num_classes = 4
    # min_values = [-9999, 250, 500, 750]
    # max_values = [250, 500, 750, -9999]
    # factor = 1.0

    #
    # main(node_file, ele_file,epoch=epoch,lr=lr,hidden_channels=hidden_channels,
    #      num_classes=num_classes,min_values=min_values,max_values=max_values,
    #      result_dir=result_dir,factor=factor)

