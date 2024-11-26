import torch
import torch.nn as nn
import torch.optim as optim
from models.cnn_model import MovieCNNModel
from models.gnn_model import MovieGraphNeuralNetwork

class MovieRecommendationTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.cnn_model = MovieCNNModel(
            input_dim=config.INPUT_DIM, 
            embedding_dim=config.EMBEDDING_DIM, 
            num_classes=config.NUM_GENRES
        ).to(self.device)
        
        self.gnn_model = MovieGraphNeuralNetwork(
            input_dim=config.INPUT_DIM,
            hidden_dim=config.HIDDEN_DIM, 
            output_dim=config.EMBEDDING_DIM
        ).to(self.device)
        
        # Loss functions and optimizers
        self.criterion = nn.CrossEntropyLoss()
        self.cnn_optimizer = optim.Adam(self.cnn_model.parameters())
        self.gnn_optimizer = optim.Adam(self.gnn_model.parameters())
    
    def train(self, train_loader, graph):
        """
        Train both CNN and GNN models
        
        Args:
            train_loader (DataLoader): Training data loader
            graph (dgl.DGLGraph): Movie graph
        """
        self.cnn_model.train()
        self.gnn_model.train()
        
        total_loss = 0
        
        for batch_features, batch_labels in train_loader:
            # Move to device
            batch_features = batch_features.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            # CNN Training
            self.cnn_optimizer.zero_grad()
            embeddings, genre_pred = self.cnn_model(batch_features)
            cnn_loss = self.criterion(genre_pred, batch_labels)
            cnn_loss.backward()
            self.cnn_optimizer.step()
            
            # GNN Training
            self.gnn_optimizer.zero_grad()
            graph_embeddings = self.gnn_model(graph, batch_features)
            # Add GNN loss computation
            
            total_loss += cnn_loss.item()
        
        return total_loss