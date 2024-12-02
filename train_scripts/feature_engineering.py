import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def load_data(filepath='data/training_data.csv'):
    return pd.read_csv(filepath)

def preprocess_data(df):
    X = df.drop(['placement', 'estimated_package'], axis=1)
    y_placement = df['placement']
    y_salary = df['estimated_package']
    
    # Define categorical and numerical features
    categorical_features = ['user_domain']
    numerical_features = ['cgpa', 'final_technical_score_percentage',
                          'soft_skills_score', 'aptitude_test_score',
                          'internship_experience']
    
    # Create transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first'), categorical_features),
            ('num', StandardScaler(), numerical_features)
        ])
    
    # Fit the preprocessor to the data
    preprocessor.fit(X)
    
    # Transform the data
    X_processed = preprocessor.transform(X)
    
    return X_processed, y_placement, y_salary, preprocessor

def save_preprocessor(preprocessor, filepath='models/preprocessor.joblib'):
    joblib.dump(preprocessor, filepath)

if __name__ == "__main__":
    df = load_data()
    X_processed, y_placement, y_salary, preprocessor = preprocess_data(df)
    save_preprocessor(preprocessor)
    print("Feature engineering and preprocessing completed.")