import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

class BollywoodRecommender:
    def __init__(self, graph, feature_extractor):
        self.graph = graph
        self.feature_extractor = feature_extractor
    
    def get_recommendations(self, base_movie, top_k=3):
        """Generate movie recommendations"""
        if base_movie not in self.graph.nodes:
            raise ValueError(f"Movie {base_movie} not found in graph")
        
        # Content-based similarity
        content_candidates = self._content_based_recommendation(base_movie)
        
        # Graph-based recommendation
        graph_candidates = self._graph_based_recommendation(base_movie)
        
        # Hybrid recommendation
        recommendations = self._merge_recommendations(
            content_candidates, 
            graph_candidates, 
            top_k
        )
        
        return recommendations
    
    def _content_based_recommendation(self, base_movie):
        """Recommend based on content similarity"""
        candidates = {}
        for movie in self.graph.nodes:
            if movie != base_movie:
                similarity = self._compute_content_similarity(base_movie, movie)
                candidates[movie] = similarity
        
        return sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    
    def _graph_based_recommendation(self, base_movie):
        """Recommend based on graph proximity"""
        # Find movies within 2 hops
        candidates = {}
        for movie in nx.single_source_shortest_path_length(
            self.graph, base_movie, cutoff=2
        ).keys():
            if movie != base_movie:
                candidates[movie] = self.graph[base_movie][movie]['weight']
        
        return sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    
    def _merge_recommendations(self, content_candidates, graph_candidates, top_k):
        """Merge and re-rank recommendations"""
        merged_candidates = {}
        
        # Weight content and graph recommendations
        for movie, score in content_candidates:
            merged_candidates[movie] = 0.6 * score
        
        for movie, score in graph_candidates:
            merged_candidates[movie] = merged_candidates.get(movie, 0) + 0.4 * score
        
        # Sort and return top K
        top_recommendations = sorted(
            merged_candidates.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k]
        
        return [movie for movie, _ in top_recommendations]

    def _compute_content_similarity(self, movie1, movie2):
        """Compute content similarity between two movies"""
        # This is a placeholder - replace with actual feature comparison
        return cosine_similarity(
            self.feature_extractor.extract_text_features([movie1]),
            self.feature_extractor.extract_text_features([movie2])
        )[0][0]