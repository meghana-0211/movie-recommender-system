import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
    RAW_DATA_PATH = 'data/bollywood_data_set.csv'
    
    def load_data(self):
        """Load raw movie dataset"""
        df = pd.read_csv(self.config.RAW_DATA_PATH)
        return df
    
    def _extract_year(self, text):
        """
        Extract year from various possible formats:
        - With parentheses: (2019)
        - With hyphen: -2012
        - Standalone year
        """
        text = str(text)
        # Try extracting year from parentheses
        parentheses_match = re.search(r'\((\d{4})\)', text)
        if parentheses_match:
            return int(parentheses_match.group(1))
        
        # Try extracting year with hyphen
        hyphen_match = re.search(r'-(\d{4})', text)
        if hyphen_match:
            return int(hyphen_match.group(1))
        
        # Try direct year match
        direct_match = re.search(r'\b(\d{4})\b', text)
        if direct_match:
            return int(direct_match.group(1))
        
        # If no year found, return NaN
        return np.nan
    
    def preprocess_data(self, df):
        """Comprehensive data preprocessing"""
        # Clean column names
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

        # Extract year from movie name
        df['year_of_release'] = df['movie_name'].apply(self._extract_year)

        # Clean and convert runtime
        df['runtime'] = df['runtime'].astype(str)  # Ensure string type
        df['runtime'] = df['runtime'].str.replace(' min', '')  # Remove 'min'
        df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce')  # Convert to numeric, invalid entries become NaN

        df['no_of_votes'] = df['no_of_votes'].astype(str).str.replace(',', '').str.strip()
    
    # Convert to numeric, handling errors
        df['no_of_votes'] = pd.to_numeric(df['no_of_votes'], errors='coerce')
    
    # Fill NaN values with 0 or another appropriate default
        df['no_of_votes'] = df['no_of_votes'].fillna(0).astype(int)
    

        # Handle missing years
        current_year = pd.Timestamp.now().year
        df['year_of_release'] = df['year_of_release'].apply(
            lambda x: x if pd.notnull(x) and 1900 <= x <= current_year else np.nan
        )
        
        # Impute missing years with median
        median_year = df['year_of_release'].median()
        df['year_of_release'] = df['year_of_release'].fillna(median_year)

        # Numeric columns for imputation
        numeric_columns = ['year_of_release', 'runtime', 'imdb_rating', 'no_of_votes']
        text_columns = ['movie_name', 'plot_description', 'director', 'actors']

        # Replace empty strings with NaN for numeric columns
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Numeric imputation using median
        numeric_imputer = SimpleImputer(strategy='median')
        df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])

        # Text column cleaning: Replace NaN with 'Unknown'
        for col in text_columns:
            df[col] = df[col].fillna('Unknown')
            df[col] = df[col].str.strip()
        
        # Feature engineering
        df['runtime_minutes'] = df['runtime'].astype(float)
        
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
        """Apply SMOTE for balancing the dataset"""
        X = df[['normalized_rating', 'normalized_votes']]
        y = df['imdb_rating']
        
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
        balanced_df['imdb_rating'] = y_resampled
        
        return balanced_df