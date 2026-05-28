import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

from sklearn.preprocessing import StandardScaler

def train_logistic_regression(data_path, model_path):
    print("Training Logistic Regression...")
    df = pd.read_csv(data_path)
    
    # Features and Target
    exclude = ['mag', 'mag_category', 'mag_category_encoded']
    X = df.drop(columns=[col for col in exclude if col in df.columns])
    y = df['mag_category_encoded']
    
    # Drop constant columns
    X = X.loc[:, X.std() > 0]
    
    # Using the full dataset for Big Data requirements
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    # Save both model and scaler
    joblib.dump({"model": model, "scaler": scaler}, model_path)
    print(f"Model and scaler saved to {model_path}")

if __name__ == "__main__":
    DATA = "earthquake_project/data/processed/cleaned_earthquakes.csv"
    MODEL = "earthquake_project/results/models/logistic_regression.joblib"
    train_logistic_regression(DATA, MODEL)
