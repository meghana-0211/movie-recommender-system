import pandas as pd
from config import Config
from src.data_preprocessing import DataPreprocessor
from src.feature_extraction import TextFeatureExtractor
from src.graph_construction import GraphConstructor
from src.recommendation_engine import BollywoodRecommender

def main():
    # Initialize configuration
    config = Config()
    
    # Data Preprocessing
    preprocessor = DataPreprocessor(config)
    raw_data = preprocessor.load_data()
    processed_data = preprocessor.preprocess_data(raw_data)
    balanced_data = preprocessor.balance_dataset(processed_data)
    
    # Feature Extraction
    feature_extractor = TextFeatureExtractor()
    
    # Graph Construction
    graph_constructor = GraphConstructor(config)
    movie_graph = graph_constructor.create_movie_graph(balanced_data)
    graph_constructor.save_to_neo4j(movie_graph)
    
    # Recommendation Engine
    recommender = BollywoodRecommender(movie_graph, feature_extractor)
    
    # Example Recommendation
    base_movie = "3 Idiots"
    recommendations = recommender.get_recommendations(base_movie)
    
    print(f"Recommendations for {base_movie}:")
    for movie in recommendations:
        print(movie)

if __name__ == "__main__":
    main()