```mermaid

flowchart TD
    subgraph Data Pipeline
        A[Movie Dataset] --> B[Data Preprocessing]
        B --> B1[SMOTE Balancing]
        B1 --> C[Feature Extraction]
        C --> C1[CNN Feature Processing]
        C1 --> D[Neo4j Graph Database]
    end

    subgraph Graph Construction
        D --> |Movie Nodes| E[Movie Properties]
        D --> |Relationship Edges| F[Similarity Scores]
        E --> G[Genre Relations]
        E --> H[Director Relations]
        E --> I[Actor Relations]
        F --> J[Content-Based Similarity]
        F --> K[Collaborative Filtering]
    end

    subgraph Hybrid RAG Implementation
        L[User Query] --> M[Query Encoder]
        M --> M1[CNN Text Processing]
        M1 --> N[Graph Neural Network]
        N --> O[Similarity Search]
        O --> O1[Content-Based Stage]
        O1 --> O2[Collaborative Stage]
        O2 --> P[Candidate Generation]
        P --> Q[Re-ranking Module]
        Q --> Q1[LeakyReLU Activation]
        Q1 --> R[Top-3 Recommendations]
    end

    subgraph Evaluation
        S[Offline Metrics] --> T[NDCG Score]
        S --> U[Precision@K]
        S --> V[Diversity Score]
        S --> W[Cold Start Performance]
        S --> X[Sparsity Handling]
    end

    subgraph CNN Processing
        C1 --> |Feature Maps| M1
        M1 --> |1D Convolution| Q1
    end