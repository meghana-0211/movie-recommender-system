
# Configuration settings for the recommendation engine
class Config:
    # Data paths
    RAW_DATA_PATH = 'data/bollywood_data_set.csv'
    PROCESSED_DATA_PATH = 'data/processed_movies.csv'
    
    # Neo4j Database Configuration
    NEO4J_URI = 'bolt://localhost:7687'
    NEO4J_USERNAME = 'neo4j'
    NEO4J_PASSWORD = '12345678'
    
    # Model Hyperparameters
    EMBEDDING_DIM = 128
    CNN_FILTER_SIZE = 3
    GNN_LAYERS = 2
    
    # Recommendation Parameters
    TOP_K_RECOMMENDATIONS = 3
    SIMILARITY_THRESHOLD = 0.7

