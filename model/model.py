from torch_geometric.nn import SAGEConv,GATConv
from torch_geometric.utils import subgraph
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_sparse import SparseTensor

# Transformer-based Scalar Value Prediction Model: Utilising Local Attention Mechanisms
class GraphTransformerLevelPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0, activation_fn='prelu'):
        super(GraphTransformerLevelPredictor, self).__init__()

        self.heads = heads
        self.embed_dim = hidden_channels  

        # Graph convolutional layer, employing GATConv for node feature fusion
        self.conv1 = GATConv(in_channels, self.embed_dim, heads=heads)  
        self.conv2 = GATConv(self.embed_dim * heads, self.embed_dim, heads=heads)  

        # Transformer-related layers
        self.attn1 = nn.MultiheadAttention(embed_dim=self.embed_dim * heads, num_heads=heads, dropout=dropout)
        self.attn2 = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=heads, dropout=dropout)
        self.attn3 = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=1, dropout=dropout)

        # Layer normalisation
        self.norm1 = nn.LayerNorm(self.embed_dim * heads)  
        self.norm2 = nn.LayerNorm(self.embed_dim)  
        self.norm3 = nn.LayerNorm(self.embed_dim)  

        # Output layer
        self.level_predictor = nn.Linear(self.embed_dim, out_channels)

        # Activation Function Selection
        if activation_fn == 'softplus':
            self.activation = nn.Softplus()
        else:
            self.activation = nn.PReLU()

    def forward(self, x, edge_index, edge_attr=None):
        # Step 1: Graph convolution, utilising the connectivity of the graph to facilitate information propagation.
        x = self.conv1(x, edge_index)  # First Layer Graph Convolution
        x = F.relu(x) 
        x = self.conv2(x, edge_index)  # Second Layer Graph Convolution
        x = F.relu(x)  

        # Step 2: Transformer Layer
        # Adapt node features to a format suitable for multi-head attention
        x = x.view(-1, self.heads, self.embed_dim) 

        # Step 3: Employing Local Attention
        x = self._apply_local_attention(x, edge_index)  # Constraining local attention based on edge connectivity relationships

        # First Transformer Layer (Local Attention)
        x = self.norm1(x.flatten(1))  # Merge the heads and embed_dim dimensions for LayerNorm
        x = x.unsqueeze(1)  
        attn_output1, _ = self.attn1(x, x, x)  # Self-attention computation
        x = x + attn_output1  # Residual connection
        x = F.dropout(x, p=self.dropout, training=self.training) 

        # Second Layer Transformer (Local Attention)
        x = self.norm2(x)
        attn_output2, _ = self.attn2(x, x, x)
        x = x + attn_output2 
        x = F.dropout(x, p=self.dropout, training=self.training)  

        
        x = self.norm3(x)
        attn_output3, _ = self.attn3(x, x, x)  
        x = x + attn_output3  
        x = F.dropout(x, p=self.dropout, training=self.training)  

        # Step 4: Output Layer
        x = x.permute(1, 0, 2)  
        x = x.flatten(1)  
        level_output = self.level_predictor(x).squeeze(-1)  #  Output the predicted Level value
        return level_output

    def _apply_local_attention(self, x, edge_index):
        """
        Constrain local attention based on edge connectivity relationships.
        Input x is the node feature matrix, and edge_index is the graph's edge connectivity information.
        """
        # Obtain the connection relationships of edges
        row, col = edge_index

        # Computing Attention Constraints Using Adjacency Matrices
        num_nodes = x.size(0)

        # Create a sparse attention mask
        attn_mask = SparseTensor(row=row, col=col, value=torch.ones(row.size(0), device=x.device), sparse_sizes=(num_nodes, num_nodes))

        # Manual calculation of weighted attention
        x = x.unsqueeze(1) 

        # Perform matrix multiplication using sparse matrices to avoid conversion to dense matrices.
        edge_weights = attn_mask.matmul(x.squeeze(1))  
        attn_output = edge_weights

        return attn_output  



# Transformer-based Scalar Value Prediction Model: Employing a Global Attention Mechanism
class GraphTransformerLevelPredictor_full_AT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0, activation_fn='prelu'):
        super(GraphTransformerLevelPredictor_full_AT, self).__init__()

        self.heads = heads
        self.embed_dim = hidden_channels  

        # Graph convolutional layer, employing GATConv for node feature fusion
        self.conv1 = GATConv(in_channels, self.embed_dim, heads=heads)
        self.conv2 = GATConv(self.embed_dim * heads, self.embed_dim, heads=heads)

        # Transformer-related layers
        self.attn1 = nn.MultiheadAttention(embed_dim=self.embed_dim * heads, num_heads=heads, dropout=dropout)
        self.attn2 = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=heads, dropout=dropout)
        self.attn3 = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=1, dropout=dropout)

        # Layer normalisation
        self.norm1 = nn.LayerNorm(self.embed_dim * heads)  
        self.norm2 = nn.LayerNorm(self.embed_dim) 
        self.norm3 = nn.LayerNorm(self.embed_dim)  

        # Output layer
        self.level_predictor = nn.Linear(self.embed_dim, out_channels)

        # Activation Function Selection
        if activation_fn == 'softplus':
            self.activation = nn.Softplus()
        else:
            self.activation = nn.PReLU()

    def forward(self, x, edge_index, edge_attr=None):
        # Step 1: Graph convolution, utilising the connectivity of the graph to facilitate information propagation.
        x = self.conv1(x, edge_index) 
        x = F.relu(x) 
        x = self.conv2(x, edge_index) 
        x = F.relu(x)  

        # Step 2: Transformer Layer
        # Adjust the dimensions of the GATConv output to accommodate the multi-head attention input.
        x = x.view(-1, self.heads, self.embed_dim)  
        # The first Transformer layer
        x = self.norm1(x.flatten(1))  # Merge the heads and embed_dim dimensions for LayerNorm
        # Adjust the shape of x to (seq_len, batch_size, embed_dim)
        x = x.unsqueeze(1)  
        attn_output1, _ = self.attn1(x, x, x) 
        x = x + attn_output1  
        x = F.dropout(x, p=self.dropout, training=self.training)  

        # Step 3: Second Layer Transformer
        x = self.norm2(x)
        attn_output2, _ = self.attn2(x, x, x)  
        x = x + attn_output2  
        x = F.dropout(x, p=self.dropout, training=self.training)  

        # Step 4: Third Layer Transformer
        x = self.norm3(x)
        attn_output3, _ = self.attn3(x, x, x)  
        x = x + attn_output3 
        x = F.dropout(x, p=self.dropout, training=self.training)  

        # Step 5: Output Layer
        x = x.permute(1, 0, 2)  
        x = x.flatten(1)  
        level_output = self.level_predictor(x).squeeze(-1)  # Output the predicted Level value
        return level_output

# Scalar Value Prediction Model Based on GAT
class GATLevelPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels,  heads=2, dropout=0, activation_fn='prelu'):

        super(GATLevelPredictor, self).__init__()
        self.heads = heads
        self.embed_dim = hidden_channels
        self.dropout = dropout

        # Graph convolutional layer, employing GATConv for node feature fusion
        self.conv1 = GATConv(in_channels, self.embed_dim, heads=heads, dropout=dropout)  
        self.conv2 = GATConv(self.embed_dim * heads, self.embed_dim, heads=heads, dropout=dropout) 
        # self.conv2 = GATConv(self.embed_dim * heads, self.embed_dim, heads=heads, dropout=dropout)  
        # self.conv3 = GATConv(self.embed_dim * heads, self.embed_dim, heads=heads, dropout=dropout)  
        # self.level_predictor = nn.Linear(self.embed_dim * heads, 1)


        # Modify the third layer GATConv by setting concat=False to achieve average pooling.
        self.conv3 = GATConv(
            self.embed_dim * heads,  
            self.embed_dim,          
            heads=heads,
            dropout=dropout,
            concat=False 
        )
        # The output layer requires adjustment of the input dimension (from embed_dim * heads to embed_dim).
        self.level_predictor = nn.Linear(self.embed_dim, 1) 

        # Activation Function Selection
        if activation_fn == 'softplus':
            self.activation = nn.Softplus()
        else:
            self.activation = nn.PReLU()

    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward propagation function.
        """
        # First Layer GATConv
        x = self.conv1(x, edge_index)  
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Second Layer GATConv
        x = self.conv2(x, edge_index)  
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Third Layer GATConv
        x = self.conv3(x, edge_index) 
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Output layer: Predicts the Level value
        level_output = self.level_predictor(x).squeeze(-1)  

        return level_output

# Scalar Value Prediction Model Based on GAT——K-Neighbourhood Approach
class KHopGATLevelPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0, K=2, activation_fn='prelu'):
        """
        LevelPredictor model for GAT based on K-neighbourhoods.
        """
        super(KHopGATLevelPredictor, self).__init__()

        self.heads = heads
        self.embed_dim = hidden_channels  # Input dimensions of the GAT
        self.K = K
        self.dropout = dropout

        # Graph convolutional layer, employing GATConv for node feature fusion
        self.conv1 = GATConv(in_channels, self.embed_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(self.embed_dim * heads, self.embed_dim, heads=heads, dropout=dropout)

        # Output layer (for predicting level values)
        self.level_predictor = nn.Linear(self.embed_dim, out_channels)

        # Activation Function Selection
        if activation_fn == 'softplus':
            self.activation = nn.Softplus()
        else:
            self.activation = nn.PReLU()

    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward propagation function.
        """
        # Obtain the subgraph of the K-order neighbourhood
        subgraph_edge_index = self.get_khop_neighbors(edge_index, x.size(0), self.K)

        # First Layer GATConv
        x = self.conv1(x, subgraph_edge_index)  
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Second Layer GATConv
        x = self.conv2(x, subgraph_edge_index)  
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Output layer: Predicts the Level value
        level_output = self.level_predictor(x).squeeze(-1)  

        return level_output

    def get_khop_neighbors(self, edge_index, num_nodes, K):
        """
        Obtain the edge indices of the subgraph within the K-order neighbourhood.
        """

        node_set = torch.arange(num_nodes).to(edge_index.device)
        subgraph_edge_index, _ = subgraph(node_set, edge_index, num_nodes=num_nodes)
        
        # Perform K iterations to obtain the K-order neighbourhood.
        for _ in range(K - 1):  
            subgraph_edge_index, _ = subgraph(subgraph_edge_index[0], subgraph_edge_index, num_nodes=num_nodes)

        return subgraph_edge_index


# Scalar value prediction based on SAGE
class LevelPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=64, activation_fn='prelu', dropout=0.0):
        """
        Initialise the LevelPredictor model
        """
        super(LevelPredictor, self).__init__()

        # Defining the Graph Convolution Layer
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr='mean')
        self.conv3 = SAGEConv(hidden_channels, out_channels, aggr='mean')

        # Define the linear output layer
        self.level_predictor = torch.nn.Linear(out_channels, 1)

        # Activation Function Selection
        if activation_fn == 'softplus':
            self.activation = torch.nn.Softplus()
        else:
            self.activation = torch.nn.PReLU()

        # dropout
        self.dropout = dropout

    def forward(self, x, edge_index):
        """
        Forward propagation of the model
        """
        # First graph convolution
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        if self.dropout > 0:
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)

        # Second graph convolution
        x = self.conv2(x, edge_index)
        x = self.activation(x)
        if self.dropout > 0:
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)

        # Third graph convolution
        x = self.conv3(x, edge_index)

        # Output layer
        level_output = self.level_predictor(x).squeeze(-1)

        return level_output


# Lithological Classification Based on SAGE
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

# GraphSage-based Rock Unit Prediction Model – Enabling Parameter Modification
class RockUnitPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=64, num_classes=4, dropout=0.0):
        super(RockUnitPredictor, self).__init__()
        self.conv1 = SAGEConv(in_channels+1 , hidden_channels,aggr='mean')  # the level feature is added.
        self.conv2 = SAGEConv(hidden_channels, hidden_channels,aggr='mean')
        self.conv3 = SAGEConv(hidden_channels, out_channels,aggr='mean')
        self.rock_unit_classifier = torch.nn.Linear(out_channels, num_classes)  # No need to add level
        self.prelu = torch.nn.PReLU()

    def forward(self, x,  edge_index):
        x = self.conv1(x, edge_index)
        x = self.prelu(x)
        x = self.conv2(x, edge_index)
        x = self.prelu(x)
        x = self.conv3(x, edge_index)

        # Classifier for rock unit prediction
        rock_unit_output = self.rock_unit_classifier(x)
        return rock_unit_output


class GATRockPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels,  num_classes, heads=2, dropout=0, activation_fn='prelu'):
        """
        A lithological classification model based on GAT.
        """
        super(GATRockPredictor, self).__init__()

        self.heads = heads
        self.embed_dim = hidden_channels  
        self.dropout = dropout

        # Graph convolutional layer, employing GATConv for node feature fusion
        self.conv1 = GATConv(in_channels, self.embed_dim, heads=heads, dropout=dropout)  
        self.conv2 = GATConv(self.embed_dim * heads, self.embed_dim, heads=heads, dropout=dropout)  
        self.conv3 = GATConv(self.embed_dim * heads, self.embed_dim, heads=heads, dropout=dropout)  

        # Modify the third layer GATConv by setting concat=False to achieve average pooling.
        self.conv3 = GATConv(
            self.embed_dim * heads,  # Input dimension
            self.embed_dim,  # Output dimension
            heads=heads,
            dropout=dropout,
            concat=False  # Key modifications: Disable stitching, enable averaging
        )
        # The output layer requires adjustment of the input dimension (from embed_dim * heads to embed_dim).
        self.rock_classifier = nn.Linear(self.embed_dim, num_classes)

        # Activation Function Selection
        if activation_fn == 'softplus':
            self.activation = nn.Softplus()
        else:
            self.activation = nn.PReLU()

    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward propagation function.
        """
        # First Layer GATConv
        x = self.conv1(x, edge_index)  # (num_nodes, embed_dim * heads)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Second Layer GATConv
        x = self.conv2(x, edge_index)  # (num_nodes, embed_dim * heads)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Third Layer GATConv
        x = self.conv3(x, edge_index)  # (num_nodes, embed_dim * heads)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Output layer: Lithological classification
        rock_unit_output = self.rock_classifier(x)  #  (num_nodes, num_classes)


        return rock_unit_output
