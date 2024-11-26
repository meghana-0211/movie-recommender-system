import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer for Movie Recommendation
    """
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        # Learnable weight matrices
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # Attention mechanism parameters
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leaky_relu = nn.LeakyReLU(self.alpha)
    
    def forward(self, graph, features):
        """
        Compute graph attention
        
        Args:
            graph (dgl.DGLGraph): Input graph
            features (torch.Tensor): Node features
        
        Returns:
            torch.Tensor: Updated node representations
        """
        graph = graph.local_var()
        
        # Linear transformation of features
        h = torch.matmul(features, self.W)
        
        # Compute attention coefficients
        graph.ndata['h'] = h
        graph.apply_edges(self.edge_attention)
        graph.edata['a'] = F.softmax(graph.edata['a'], dim=1)
        
        # Message passing
        graph.edata['a'] = F.dropout(graph.edata['a'], self.dropout)
        graph.update_all(fn.u_mul_e('h', 'a', 'm'), fn.sum('m', 'h'))
        
        return graph.ndata['h']
    
    def edge_attention(self, edges):
        """
        Compute edge attention scores
        """
        concat_features = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
        edge_attention = self.leaky_relu(torch.matmul(concat_features, self.a))
        return {'a': edge_attention}

class MovieGraphNeuralNetwork(nn.Module):
    """
    Graph Neural Network for Movie Recommendation
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(MovieGraphNeuralNetwork, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(GraphAttentionLayer(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(GraphAttentionLayer(hidden_dim, hidden_dim))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # Dropout and activation
        self.dropout = nn.Dropout(0.6)
        self.leaky_relu = nn.LeakyReLU(0.2)
    
    def forward(self, graph, features):
        """
        Forward pass through graph neural network
        
        Args:
            graph (dgl.DGLGraph): Input movie graph
            features (torch.Tensor): Initial node features
        
        Returns:
            torch.Tensor: Final node embeddings
        """
        x = features
        
        # Graph convolution layers
        for layer in self.layers:
            x = layer(graph, x)
            x = self.leaky_relu(x)
            x = self.dropout(x)
        
        # Final output layer
        x = self.output_layer(x)
        
        return x

def create_movie_graph(movie_data):
    """
    Create a DGL graph from movie data
    
    Args:
        movie_data (pd.DataFrame): Movie dataset
    
    Returns:
        dgl.DGLGraph: Graph representation of movies
    """
    import dgl
    import torch
    
    # Create graph based on movie similarities
    num_movies = len(movie_data)
    
    # Create edges based on similarity/genre/actor connections
    src_nodes = []
    dst_nodes = []
    
    # Example: Connect movies with similar genres
    for i in range(num_movies):
        for j in range(i+1, num_movies):
                src_nodes.append(i)
                dst_nodes.append(j)
    
    # Create graph
    graph = dgl.graph((src_nodes, dst_nodes), num_nodes=num_movies)
    
    # Add node features
    node_features = torch.tensor(movie_data[['rating', 'votes']].values, dtype=torch.float)
    graph.ndata['features'] = node_features
    
    return graph
