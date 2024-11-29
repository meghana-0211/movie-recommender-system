# Bollywood Movie Recommendation Engine

## Project Overview

This project implements an advanced hybrid Retrieval-Augmented Generation (RAG) recommendation system specifically designed for Bollywood movies. The system leverages sophisticated machine learning techniques, graph databases, and cultural nuances to provide personalized movie recommendations.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Dataset](#dataset)
5. [Architecture](#architecture)
6. [Implementation Details](#implementation-details)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Cultural Awareness](#cultural-awareness)
9. [Usage](#usage)
10. [Contribution Guidelines](#contribution-guidelines)
11. [License](#license)

## Project Structure

```
bollywood-recommendation-engine/
│
├── data/
│   ├── raw_dataset.csv
│   └── preprocessed_dataset.csv
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_extraction.py
│   ├── graph_construction.py
│   ├── recommendation_engine.py
│   └── evaluation.py
│
├── models/
│   ├── cnn_model.py
│   └── gnn_model.py
│
├── notebooks/
│   └── exploration.ipynb
│
├── requirements.txt
├── README.md
└── main.py
```

## Prerequisites

- Python 3.8+
- Neo4j Graph Database (Community Edition)
- Libraries:
  - numpy
  - pandas
  - scikit-learn
  - torch
  - transformers
  - neo4j-python-driver
  - SMOTE from imbalanced-learn
  - NetworkX
  - Indic-BERT

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/bollywood-recommendation-engine.git
cd bollywood-recommendation-engine
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up Neo4j Database:
- Download Neo4j Community Edition
- Create a new database
- Update connection credentials in `config.py`

## Dataset

### Source
- Dataset: Bollywood Movies Dataset
- Source: Kaggle (https://www.kaggle.com/datasets/mustafaanandwala/10000-bollywood-dataset)
- Total Movies: 10,000+
- Features:
  - Movie Name
  - Year of Release
  - Runtime
  - IMDB Rating
  - Number of Votes
  - Plot Description
  - Director
  - Actors

### Preprocessing Steps
- Remove duplicate entries
- Handle missing values
- Encode categorical variables
- Apply SMOTE for balanced representation
- Extract text features using CNN
- Generate movie embeddings

## Architecture
![image](https://github.com/user-attachments/assets/7cfa4d3c-d13b-4912-b1e3-df38bcc4e0e3)


### Data Pipeline
1. **Data Preprocessing**
   - Clean and format dataset
   - Encode categorical variables
   - Apply SMOTE balancing

2. **Feature Extraction**
   - Extract metadata features
   - Use CNN for text feature processing
   - Store in Neo4j graph database

### Graph Construction
- **Nodes**: Movies, Actors, Directors, Genres
- **Edges**: Relationships like acted_in, directed_by, has_genre
- **Similarity Computation**
  - Cosine similarity between movie embeddings
  - Content-based and collaborative filtering

### Hybrid RAG Implementation
- Query encoding using CNN
- Graph Neural Network for embedding learning
- Two-stage recommendation:
  1. Content-based recommendation
  2. Collaborative filtering
- Re-ranking with LeakyReLU activation

## Evaluation Metrics

1. **NDCG Score**: Ranking quality of recommendations
2. **Precision@K**: Proportion of relevant recommendations
3. **Diversity Score**: Recommendation variety
4. **Cold Start Performance**: Handling new users/movies
5. **Sparsity Handling**: Performance in limited data scenarios

## Cultural Awareness

### Special Features
- Indic-BERT integration for language understanding
- Regional movie industry weighting
- Seasonal context consideration
- Cultural trope recognition

## Usage

### Recommendation Example
```python
from recommendation_engine import MovieRecommender

recommender = MovieRecommender()
recommended_movies = recommender.get_recommendations("3 Idiots")
print(recommended_movies)
# Output: Top 3 similar movies based on content and collaborative filtering
```

## Model Performance Snapshot
- Average NDCG Score: 0.85
- Precision@3: 0.75
- Diversity Score: 0.68
- Cold Start Performance: 0.62

## Output Image
![Screenshot 2024-11-30 014534](https://github.com/user-attachments/assets/36d9c6e9-1fe6-43c6-8665-f4a5cc210c79)


## Contribution Guidelines

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

### Contribution Areas
- Improve cultural context understanding
- Enhance feature extraction techniques
- Optimize graph neural network
- Add more evaluation metrics

## Future Roadmap
- Multi-language support
- Real-time recommendation updating
- User preference learning
- Expanded cultural context modeling

## Limitations
- Dependent on dataset quality
- Potential bias in cultural recommendations
- Computational complexity of graph operations

## License
MIT License

## References
https://eprints.uad.ac.id/65078/1/1-Movie%20Recommender%20System%20with%20Cascade%20Hybrid%20Filtering%20Using%20Convolutional%20Neural%20Network.pdf

## Acknowledgments
- Kaggle for the dataset
- Indic-BERT for language processing
- Neo4j for graph database
- Open-source machine learning libraries

---

**Note**: This is an experimental recommendation system. Recommendations are generated based on complex machine learning algorithms and may not always perfectly match user preferences.
