import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

def train_xgboost(data_path, model_path):
    print("Training XGBoost...")
    df = pd.read_csv(data_path)
    
    # Features and Target
    exclude = ['mag', 'mag_category', 'mag_category_encoded']
    X = df.drop(columns=[col for col in exclude if col in df.columns])
    y = df['mag_category_encoded']
    
    # Drop constant columns
    X = X.loc[:, X.std() > 0]
    
    # Using the full dataset for Big Data requirements
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBClassifier(n_estimators=50, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    DATA = "earthquake_project/data/processed/cleaned_earthquakes.csv"
    MODEL = "earthquake_project/results/models/gradient_boosting.joblib"
    train_xgboost(DATA, MODEL)
