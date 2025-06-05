from torch_geometric.nn import SAGEConv,GATConv
from torch_geometric.utils import subgraph
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_sparse import SparseTensor
# 网络模型

## 基于Transformer的标量值预测模型——使用局部注意力机制
class GraphTransformerLevelPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0, activation_fn='prelu'):
        super(GraphTransformerLevelPredictor, self).__init__()

        self.heads = heads
        self.embed_dim = hidden_channels  # 我们选择更大的维度作为 Transformer 的输入维度

        # 图卷积层，利用 GATConv 进行节点特征融合
        self.conv1 = GATConv(in_channels, self.embed_dim, heads=heads)  # 输出维度是 heads * embed_dim
        self.conv2 = GATConv(self.embed_dim * heads, self.embed_dim, heads=heads)  # 同理

        # Transformer 相关层
        self.attn1 = nn.MultiheadAttention(embed_dim=self.embed_dim * heads, num_heads=heads, dropout=dropout)
        self.attn2 = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=heads, dropout=dropout)
        self.attn3 = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=1, dropout=dropout)

        # 层归一化
        self.norm1 = nn.LayerNorm(self.embed_dim * heads)  # 注意归一化的维度应该是 embed_dim * heads
        self.norm2 = nn.LayerNorm(self.embed_dim)  # 对应第二层 Transformer
        self.norm3 = nn.LayerNorm(self.embed_dim)  # 对应第三层 Transformer

        # 输出层
        self.level_predictor = nn.Linear(self.embed_dim, out_channels)

        # 激活函数选择
        if activation_fn == 'softplus':
            self.activation = nn.Softplus()
        else:
            self.activation = nn.PReLU()

    def forward(self, x, edge_index, edge_attr=None):
        # Step 1: 图卷积，利用图的连接关系进行信息传递
        x = self.conv1(x, edge_index)  # 第一层图卷积
        x = F.relu(x)  # 激活函数
        x = self.conv2(x, edge_index)  # 第二层图卷积
        x = F.relu(x)  # 激活函数

        # Step 2: Transformer 层
        # 将节点特征调整为适合多头注意力的格式
        x = x.view(-1, self.heads, self.embed_dim)  # 维度是 [num_nodes, heads, embed_dim]

        # Step 3: 使用局部注意力（根据节点的连接关系）
        x = self._apply_local_attention(x, edge_index)  # 根据边的连接关系限制局部注意力

        # 第一个 Transformer 层（局部注意力）
        x = self.norm1(x.flatten(1))  # 合并 heads 和 embed_dim 维度进行 LayerNorm
        x = x.unsqueeze(1)  # (num_nodes, 1, embed_dim) 这里的 1 表示 batch_size
        attn_output1, _ = self.attn1(x, x, x)  # 自注意力计算
        x = x + attn_output1  # 残差连接
        x = F.dropout(x, p=self.dropout, training=self.training)  # 在残差连接之后应用 dropout

        # 第二层 Transformer（局部注意力）
        x = self.norm2(x)
        attn_output2, _ = self.attn2(x, x, x)  # 自注意力计算
        x = x + attn_output2  # 残差连接
        x = F.dropout(x, p=self.dropout, training=self.training)  # 在残差连接之后应用 dropout

        # 第三层 Transformer（局部注意力）
        x = self.norm3(x)
        attn_output3, _ = self.attn3(x, x, x)  # 自注意力计算
        x = x + attn_output3  # 残差连接
        x = F.dropout(x, p=self.dropout, training=self.training)  # 在残差连接之后应用 dropout

        # Step 4: 输出层
        x = x.permute(1, 0, 2)  # 转换回 [num_nodes, heads, embed_dim]
        x = x.flatten(1)  # 扁平化为 (num_nodes, embed_dim * heads)
        level_output = self.level_predictor(x).squeeze(-1)  # 输出预测的 Level 值
        return level_output

    def _apply_local_attention(self, x, edge_index):
        """
        根据边的连接关系限制局部注意力。
        输入 x 是节点特征矩阵，edge_index 是图的边连接关系。
        """
        # 获取边的连接关系
        row, col = edge_index

        # 使用邻接矩阵限制注意力的计算
        num_nodes = x.size(0)

        # 创建一个稀疏的注意力掩码
        attn_mask = SparseTensor(row=row, col=col, value=torch.ones(row.size(0), device=x.device), sparse_sizes=(num_nodes, num_nodes))

        # 手动计算加权注意力
        x = x.unsqueeze(1)  # 增加 batch 维度

        # 使用稀疏矩阵进行乘法，避免转为稠密矩阵
        # 注意：不需要转为稠密矩阵，而是直接在稀疏矩阵上操作
        edge_weights = attn_mask.matmul(x.squeeze(1))  # 使用稀疏矩阵乘法
        attn_output = edge_weights

        return attn_output  # 返回经过局部注意力计算的节点特征



##基于Transformer的标量值预测模型——使用全局注意力机制
class GraphTransformerLevelPredictor_full_AT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0, activation_fn='prelu'):
        super(GraphTransformerLevelPredictor_full_AT, self).__init__()

        self.heads = heads
        self.embed_dim = hidden_channels  # 假设我们选择更大的维度作为 Transformer 的输入维度

        # 图卷积层，利用 GATConv 进行节点特征融合
        self.conv1 = GATConv(in_channels, self.embed_dim, heads=heads)  # 输出维度是 heads * embed_dim
        self.conv2 = GATConv(self.embed_dim * heads, self.embed_dim, heads=heads)  # 同理

        # Transformer 相关层
        self.attn1 = nn.MultiheadAttention(embed_dim=self.embed_dim * heads, num_heads=heads, dropout=dropout)
        self.attn2 = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=heads, dropout=dropout)
        self.attn3 = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=1, dropout=dropout)

        # 层归一化
        self.norm1 = nn.LayerNorm(self.embed_dim * heads)  # 注意归一化的维度应该是 embed_dim * heads
        self.norm2 = nn.LayerNorm(self.embed_dim)  # 对应第二层 Transformer
        self.norm3 = nn.LayerNorm(self.embed_dim)  # 对应第三层 Transformer

        # 输出层
        self.level_predictor = nn.Linear(self.embed_dim, out_channels)

        # 激活函数选择
        if activation_fn == 'softplus':
            self.activation = nn.Softplus()
        else:
            self.activation = nn.PReLU()

    def forward(self, x, edge_index, edge_attr=None):
        # Step 1: 图卷积，利用图的连接关系进行信息传递
        x = self.conv1(x, edge_index)  # 第一层图卷积
        x = F.relu(x)  # 激活函数
        x = self.conv2(x, edge_index)  # 第二层图卷积
        x = F.relu(x)  # 激活函数

        # Step 2: Transformer 层
        # 调整 GATConv 输出的维度，以适应多头注意力的输入
        x = x.view(-1, self.heads, self.embed_dim)  # 维度是 [num_nodes, heads, embed_dim]
        # 第一个 Transformer 层
        x = self.norm1(x.flatten(1))  # 合并 heads 和 embed_dim 维度进行 LayerNorm
        # 调整 x 的形状为 (seq_len, batch_size, embed_dim)
        x = x.unsqueeze(1)  # (num_nodes, 1, embed_dim) 这里的 1 表示 batch_size
        attn_output1, _ = self.attn1(x, x, x)  # 自注意力计算
        x = x + attn_output1  # 残差连接
        x = F.dropout(x, p=self.dropout, training=self.training)  # 在残差连接之后应用 dropout

        # Step 3: 第二层 Transformer
        x = self.norm2(x)
        attn_output2, _ = self.attn2(x, x, x)  # 自注意力计算
        x = x + attn_output2  # 残差连接
        x = F.dropout(x, p=self.dropout, training=self.training)  # 在残差连接之后应用 dropout

        # Step 4: 第三层 Transformer
        x = self.norm3(x)
        attn_output3, _ = self.attn3(x, x, x)  # 自注意力计算
        x = x + attn_output3  # 残差连接
        x = F.dropout(x, p=self.dropout, training=self.training)  # 在残差连接之后应用 dropout

        # Step 5: 输出层
        x = x.permute(1, 0, 2)  # 转换回 [num_nodes, heads, embed_dim]
        x = x.flatten(1)  # 扁平化为 (num_nodes, embed_dim * heads)
        level_output = self.level_predictor(x).squeeze(-1)  # 输出预测的 Level 值
        return level_output

##基于GAT的标量值预测模型
class GATLevelPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels,  heads=2, dropout=0, activation_fn='prelu'):
        """
        基于 GAT 的 LevelPredictor 模型。

        参数：
        - in_channels (int): 输入节点特征的维度。
        - hidden_channels (int): 隐藏层特征的维度。
        - heads (int): 注意力头的数量。
        - dropout (float): Dropout 概率。
        - activation_fn (str): 激活函数选择，'prelu' 或 'softplus'。
        """
        super(GATLevelPredictor, self).__init__()

        self.heads = heads
        self.embed_dim = hidden_channels  # 假设我们选择更大的维度作为模型的输入维度
        self.dropout = dropout

        # 图卷积层，利用 GATConv 进行节点特征融合
        self.conv1 = GATConv(in_channels, self.embed_dim, heads=heads, dropout=dropout)  # 输出维度是 heads * embed_dim
        self.conv2 = GATConv(self.embed_dim * heads, self.embed_dim, heads=heads, dropout=dropout)  # 同理
        # self.conv2 = GATConv(self.embed_dim * heads, self.embed_dim, heads=heads, dropout=dropout)  # 同理
        # self.conv3 = GATConv(self.embed_dim * heads, self.embed_dim, heads=heads, dropout=dropout)  # 同理
        # # 输出层（用于预测 level 值）
        # self.level_predictor = nn.Linear(self.embed_dim * heads, 1)


        # 修改第三层 GATConv，设置 concat=False 以实现平均聚合
        self.conv3 = GATConv(
            self.embed_dim * heads,  # 输入维度（前两层的输出是拼接后的 heads * embed_dim）
            self.embed_dim,          # 输出维度（平均后保持 embed_dim）
            heads=heads,
            dropout=dropout,
            concat=False  # 关键修改：关闭拼接，启用平均
        )
        # 输出层需要调整输入维度（从 embed_dim * heads 改为 embed_dim）
        self.level_predictor = nn.Linear(self.embed_dim, 1)  # 输入维度改为 embed_dim

        # 激活函数选择
        if activation_fn == 'softplus':
            self.activation = nn.Softplus()
        else:
            self.activation = nn.PReLU()

    def forward(self, x, edge_index, edge_attr=None):
        """
        前向传播函数。

        参数：
        - x (Tensor): 节点特征，形状为 (num_nodes, in_channels)。
        - edge_index (Tensor): 边索引，形状为 (2, num_edges)。
        - edge_attr (Tensor, optional): 边的特征，形状为 (num_edges, num_edge_features)。默认值为 None。

        返回：
        - level_output (Tensor): 预测的 level 值，形状为 (num_nodes,)。
        """
        # 第一层 GATConv
        x = self.conv1(x, edge_index)  # 输出形状: (num_nodes, embed_dim * heads)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 第二层 GATConv
        x = self.conv2(x, edge_index)  # 输出形状: (num_nodes, embed_dim * heads)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 第三层 GATConv
        x = self.conv3(x, edge_index)  # 输出形状: (num_nodes, embed_dim * heads)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 输出层：预测 Level 值
        level_output = self.level_predictor(x).squeeze(-1)  # 输出形状: (num_nodes,)

        return level_output

##基于GAT的标量值预测模型_——基于K阶邻域
class KHopGATLevelPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0, K=2, activation_fn='prelu'):
        """
        基于 K 阶邻域的 GAT 的 LevelPredictor 模型。

        参数：
        - in_channels (int): 输入节点特征的维度。
        - hidden_channels (int): 隐藏层特征的维度。
        - out_channels (int): 输出特征的维度（通常为 1，表示预测的 level 值）。
        - heads (int): 注意力头的数量。
        - dropout (float): Dropout 概率。
        - K (int): 邻域的阶数，即考虑多少层邻居节点。
        - activation_fn (str): 激活函数选择，'prelu' 或 'softplus'。
        """
        super(KHopGATLevelPredictor, self).__init__()

        self.heads = heads
        self.embed_dim = hidden_channels  # GAT 的输入维度
        self.K = K
        self.dropout = dropout

        # 图卷积层，利用 GATConv 进行节点特征融合
        self.conv1 = GATConv(in_channels, self.embed_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(self.embed_dim * heads, self.embed_dim, heads=heads, dropout=dropout)

        # 输出层（用于预测 level 值）
        self.level_predictor = nn.Linear(self.embed_dim, out_channels)

        # 激活函数选择
        if activation_fn == 'softplus':
            self.activation = nn.Softplus()
        else:
            self.activation = nn.PReLU()

    def forward(self, x, edge_index, edge_attr=None):
        """
        前向传播函数。

        参数：
        - x (Tensor): 节点特征，形状为 (num_nodes, in_channels)。
        - edge_index (Tensor): 边索引，形状为 (2, num_edges)。
        - edge_attr (Tensor, optional): 边的特征，形状为 (num_edges, num_edge_features)。默认值为 None。

        返回：
        - level_output (Tensor): 预测的 level 值，形状为 (num_nodes,)。
        """
        # 获取 K 阶邻域的子图
        subgraph_edge_index = self.get_khop_neighbors(edge_index, x.size(0), self.K)

        # 第一层 GATConv
        x = self.conv1(x, subgraph_edge_index)  # 输出形状: (num_nodes, embed_dim * heads)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 第二层 GATConv
        x = self.conv2(x, subgraph_edge_index)  # 输出形状: (num_nodes, embed_dim * heads)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 输出层：预测 Level 值
        level_output = self.level_predictor(x).squeeze(-1)  # 输出形状: (num_nodes,)

        return level_output

    def get_khop_neighbors(self, edge_index, num_nodes, K):
        """
        获取 K 阶邻域的子图的边索引。

        参数：
        - edge_index (Tensor): 图的边索引，形状为 (2, num_edges)。
        - num_nodes (int): 节点数。
        - K (int): 邻域的阶数。

        返回：
        - subgraph_edge_index (Tensor): K 阶邻域的边索引。
        """
        # 使用 `torch_geometric.utils` 来提取 K 阶邻域的边
        # 我们可以使用 `torch_geometric.utils.k_hop_subgraph` 或 `subgraph` 来获取 K 阶邻域的子图
        node_set = torch.arange(num_nodes).to(edge_index.device)
        subgraph_edge_index, _ = subgraph(node_set, edge_index, num_nodes=num_nodes)

        for _ in range(K - 1):  # 进行 K 次迭代，获取 K 阶邻域
            subgraph_edge_index, _ = subgraph(subgraph_edge_index[0], subgraph_edge_index, num_nodes=num_nodes)

        return subgraph_edge_index


## 基于SAGE进行标量值预测
class LevelPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=64, activation_fn='prelu', dropout=0.0):
        """
        初始化 LevelPredictor 模型

        参数:
        - in_channels: 输入特征的维度
        - hidden_channels: 隐藏层的通道数
        - out_channels: 输出特征的维度
        - activation_fn: 激活函数，默认为 'prelu'，可以选择 'softplus'
        - dropout: dropout 概率，默认为 0，表示不使用 dropout
        """
        super(LevelPredictor, self).__init__()

        # 定义图卷积层
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr='mean')
        self.conv3 = SAGEConv(hidden_channels, out_channels, aggr='mean')

        # 定义线性输出层
        self.level_predictor = torch.nn.Linear(out_channels, 1)

        # 激活函数选择
        if activation_fn == 'softplus':
            self.activation = torch.nn.Softplus()
        else:
            self.activation = torch.nn.PReLU()

        # dropout
        self.dropout = dropout

    def forward(self, x, edge_index):
        """
        模型的前向传播

        参数:
        - x: 节点特征
        - edge_index: 图的边索引

        返回:
        - level_output: 预测的 Level 输出
        """
        # 第一次图卷积
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        if self.dropout > 0:
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)

        # 第二次图卷积
        x = self.conv2(x, edge_index)
        x = self.activation(x)
        if self.dropout > 0:
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)

        # 第三次图卷积
        x = self.conv3(x, edge_index)

        # 输出层
        level_output = self.level_predictor(x).squeeze(-1)

        return level_output

    # 基于graphsage的Level预测模型   旧版本的graphsage
##基于SAGE进行岩性分类
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)
        self.level_predictor = torch.nn.Linear(out_channels, 1)
        self.prelu = torch.nn.PReLU()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.prelu(x)
        x = self.conv2(x, edge_index)
        x = self.prelu(x)
        x = self.conv3(x, edge_index)
        level_output = self.level_predictor(x).squeeze(-1)
        return level_output

# 基于graphsage的岩单元预测模型——提供参数修改
class RockUnitPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=64, num_classes=4, dropout=0.0):
        super(RockUnitPredictor, self).__init__()
        self.conv1 = SAGEConv(in_channels+1 , hidden_channels,aggr='mean')  # 额外加上 level 特征
        self.conv2 = SAGEConv(hidden_channels, hidden_channels,aggr='mean')
        self.conv3 = SAGEConv(hidden_channels, out_channels,aggr='mean')
        self.rock_unit_classifier = torch.nn.Linear(out_channels, num_classes)  # 不需要加 level
        self.prelu = torch.nn.PReLU()

    def forward(self, x,  edge_index):
        x = self.conv1(x, edge_index)
        x = self.prelu(x)
        x = self.conv2(x, edge_index)
        x = self.prelu(x)
        x = self.conv3(x, edge_index)

        # 分类器进行岩单元预测
        rock_unit_output = self.rock_unit_classifier(x)
        return rock_unit_output


class GATRockPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels,  num_classes, heads=2, dropout=0, activation_fn='prelu'):
        """
        基于 GAT 的岩性分类模型。

        参数：
        - in_channels (int): 输入节点特征的维度。
        - hidden_channels (int): 隐藏层特征的维度。
        - num_classes (int): 分类的类别数量（即岩性的种类）。
        - heads (int): 注意力头的数量。
        - dropout (float): Dropout 概率。
        - activation_fn (str): 激活函数选择，'prelu' 或 'softplus'。
        """
        super(GATRockPredictor, self).__init__()

        self.heads = heads
        self.embed_dim = hidden_channels  # 假设我们选择更大的维度作为模型的输入维度
        self.dropout = dropout

        # 图卷积层，利用 GATConv 进行节点特征融合
        self.conv1 = GATConv(in_channels, self.embed_dim, heads=heads, dropout=dropout)  # 输出维度是 heads * embed_dim
        self.conv2 = GATConv(self.embed_dim * heads, self.embed_dim, heads=heads, dropout=dropout)  # 同理
        # self.conv3 = GATConv(self.embed_dim * heads, self.embed_dim, heads=heads, dropout=dropout)  # 第三层 GATConv
        # # 输出层（用于岩性分类）
        # self.rock_classifier = nn.Linear(self.embed_dim * heads, num_classes)

        # 修改第三层 GATConv，设置 concat=False 以实现平均聚合
        self.conv3 = GATConv(
            self.embed_dim * heads,  # 输入维度（前两层的输出是拼接后的 heads * embed_dim）
            self.embed_dim,  # 输出维度（平均后保持 embed_dim）
            heads=heads,
            dropout=dropout,
            concat=False  # 关键修改：关闭拼接，启用平均
        )
        # 输出层需要调整输入维度（从 embed_dim * heads 改为 embed_dim）
        self.rock_classifier = nn.Linear(self.embed_dim, num_classes)

        # 激活函数选择
        if activation_fn == 'softplus':
            self.activation = nn.Softplus()
        else:
            self.activation = nn.PReLU()

    def forward(self, x, edge_index, edge_attr=None):
        """
        前向传播函数。

        参数：
        - x (Tensor): 节点特征，形状为 (num_nodes, in_channels)。
        - edge_index (Tensor): 边索引，形状为 (2, num_edges)。
        - edge_attr (Tensor, optional): 边的特征，形状为 (num_edges, num_edge_features)。默认值为 None。

        返回：
        - rock_unit_output (Tensor): 预测的岩性类别，形状为 (num_nodes, num_classes)。
        """
        # 第一层 GATConv
        x = self.conv1(x, edge_index)  # 输出形状: (num_nodes, embed_dim * heads)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 第二层 GATConv
        x = self.conv2(x, edge_index)  # 输出形状: (num_nodes, embed_dim * heads)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 第三层 GATConv
        x = self.conv3(x, edge_index)  # 输出形状: (num_nodes, embed_dim * heads)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 输出层：进行岩性分类
        rock_unit_output = self.rock_classifier(x)  # 输出形状: (num_nodes, num_classes)

        return rock_unit_output