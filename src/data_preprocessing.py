import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
    
    def load_data(self):
        """Load raw movie dataset"""
        df = pd.read_csv(self.config.RAW_DATA_PATH)
        return df
    
    def preprocess_data(self, df):
        """Comprehensive data preprocessing"""
        # Clean column names
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
        
        # Handle missing values
        numeric_columns = ['year_of_release', 'runtime', 'imdb_rating', 'no_of_votes']
        text_columns = ['movie_name', 'plot_description', 'director', 'actors']
        
        # Numeric imputation
        numeric_imputer = SimpleImputer(strategy='median')
        df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])
        
        # Text column cleaning
        for col in text_columns:
            df[col] = df[col].fillna('Unknown')
            df[col] = df[col].str.strip()
        
        # Feature engineering
        df['runtime_minutes'] = df['runtime'].str.extract('(\d+)').astype(float)
        
        # Encode categorical features
        df['directors_encoded'] = self._encode_categorical(df['director'])
        df['actors_encoded'] = self._encode_categorical(df['actors'], delimiter='|')
        
        # Normalize numeric features
        scaler = StandardScaler()
        df[['normalized_rating', 'normalized_votes']] = scaler.fit_transform(
            df[['imdb_rating', 'no_of_votes']]
        )
        
        return df
    
    def _encode_categorical(self, series, delimiter=None):
        """Encode categorical variables"""
        if delimiter:
            series = series.str.split(delimiter)
        
        mlb = MultiLabelBinarizer()
        encoded = mlb.fit_transform(series)
        return encoded
    
    def balance_dataset(self, df):
        """Apply SMOTE for balancing"""
        X = df[['normalized_rating', 'normalized_votes']]
        y = df['imdb_rating']
        
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
        balanced_df['imdb_rating'] = y_resampled
        
        return balanced_df