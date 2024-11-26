import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class TextFeatureExtractor:
    def __init__(self, model_name='ai4bharat/indic-bert'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
    def extract_text_features(self, texts):
        """Extract deep semantic features from text"""
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        return outputs.last_hidden_state.mean(dim=1)

class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=input_dim, 
            out_channels=embedding_dim, 
            kernel_size=3, 
            padding=1
        )
        self.relu = nn.LeakyReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
    
    def forward(self, x):
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.pool(x)
        return x