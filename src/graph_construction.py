from neo4j import GraphDatabase
import networkx as nx
import numpy as np

class GraphConstructor:
    def __init__(self, config):
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI, 
            auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
        )
    
    def create_movie_graph(self, movies_df):
        """Construct graph representation of movies"""
        G = nx.Graph()
        
        # Add movie nodes
        for _, movie in movies_df.iterrows():
            G.add_node(
                movie['movie_name'], 
                type='movie',
                rating=movie['imdb_rating'],
                year=movie['year_of_release']
            )
        
        # Add edges based on similarity
        similarity_matrix = self._compute_movie_similarity(movies_df)
        
        for i in range(len(movies_df)):
            for j in range(i+1, len(movies_df)):
                if similarity_matrix[i, j] > 0.7:  # Similarity threshold
                    G.add_edge(
                        movies_df.iloc[i]['movie_name'], 
                        movies_df.iloc[j]['movie_name'], 
                        weight=similarity_matrix[i, j]
                    )
        
        return G
    
    def _compute_movie_similarity(self, movies_df):
        """Compute cosine similarity between movies"""
        features = movies_df[['normalized_rating', 'normalized_votes']].values
        
        # Cosine similarity
        norm = np.linalg.norm(features, axis=1)
        similarity = np.dot(features, features.T) / (norm[:, None] * norm[None, :])
        
        return similarity
    
    def save_to_neo4j(self, graph):
        """Save graph to Neo4j database"""
        with self.driver.session() as session:
            # Clear existing data
            session.run("MATCH (n) DETACH DELETE n")
            
            # Create movie nodes
            for node, data in graph.nodes(data=True):
                session.run(
                    "CREATE (m:Movie {name: $name, rating: $rating, year: $year})",
                    name=node, 
                    rating=data.get('rating', 0), 
                    year=data.get('year', 0)
                )
            
            # Create movie edges
            for u, v, data in graph.edges(data=True):
                session.run(
                    "MATCH (a:Movie {name: $name1}), (b:Movie {name: $name2}) " 
                    "CREATE (a)-[:SIMILAR {weight: $weight}]->(b)",
                    name1=u, name2=v, weight=data.get('weight', 0)
                )