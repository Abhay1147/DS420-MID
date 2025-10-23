import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class HRDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.department_encoder = LabelEncoder()
        self.salary_encoder = LabelEncoder()
        self.is_fitted = False
    
    def fit_transform(self, df):
        """Fit and transform the data"""
        # Create a copy to avoid modifying original
        data = df.copy()
        
        # Encode categorical variables
        data['Department'] = self.department_encoder.fit_transform(data['Department'])
        data['salary'] = self.salary_encoder.fit_transform(data['salary'])
        
        # Separate features and target
        X = data.drop('left', axis=1)
        y = data['left']
        
        # Scale numerical features
        numerical_features = ['satisfaction_level', 'last_evaluation', 'number_project', 
                            'average_montly_hours', 'time_spend_company']
        X[numerical_features] = self.scaler.fit_transform(X[numerical_features])
        
        self.is_fitted = True
        return X, y
    
    def transform(self, df):
        """Transform new data using fitted preprocessor"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first")
        
        data = df.copy()
        data['Department'] = self.department_encoder.transform(data['Department'])
        data['salary'] = self.salary_encoder.transform(data['salary'])
        
        numerical_features = ['satisfaction_level', 'last_evaluation', 'number_project', 
                            'average_montly_hours', 'time_spend_company']
        data[numerical_features] = self.scaler.transform(data[numerical_features])
        
        return data

def prepare_data(df, test_size=0.2, random_state=42):
    """Complete data preparation pipeline"""
    preprocessor = HRDataPreprocessor()
    X, y = preprocessor.fit_transform(df)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, preprocessor

if __name__ == "__main__":
    # Test the preprocessor
    df = pd.read_csv('HR_data.csv')
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")