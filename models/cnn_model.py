import torch
import torch.nn as nn
import torch.nn.functional as F

class MovieCNNModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_classes):
        """
        CNN Model for Movie Feature Extraction and Representation
        
        Args:
            input_dim (int): Input feature dimension
            embedding_dim (int): Embedding dimension for movie representation
            num_classes (int): Number of movie categories/genres
        """
        super(MovieCNNModel, self).__init__()
        
        # Convolutional Layers
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, embedding_dim, kernel_size=3, padding=1)
        
        # Batch Normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(embedding_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(embedding_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Activation Functions
        self.leaky_relu = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        """
        Forward pass through the CNN
        
        Args:
            x (torch.Tensor): Input movie features
        
        Returns:
            torch.Tensor: Movie embeddings and genre predictions
        """
        # Convolutional Layers with LeakyReLU and Batch Normalization
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, kernel_size=2)
        
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=2)
        
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        
        # Dropout for regularization
        x = self.dropout(x)
        
        # Fully Connected Layers
        x = self.leaky_relu(self.fc1(x))
        genre_predictions = self.fc2(x)
        
        return x, genre_predictions

class CNNFeatureExtractor(nn.Module):
    """
    Specialized Feature Extractor using CNN
    """
    def __init__(self, input_dim, embedding_dim):
        super(CNNFeatureExtractor, self).__init__()
        
        self.extract_features = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(128, embedding_dim, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(1)
        )
    
    def forward(self, x):
        """
        Extract features from input tensor
        
        Args:
            x (torch.Tensor): Input features
        
        Returns:
            torch.Tensor: Extracted movie embedding
        """
        features = self.extract_features(x)
        return features.squeeze(-1)