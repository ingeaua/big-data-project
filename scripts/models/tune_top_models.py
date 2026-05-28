import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import time

def tune_models(data_path):
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    exclude = ['mag', 'mag_category', 'mag_category_encoded']
    X = df.drop(columns=[col for col in exclude if col in df.columns])
    y = df['mag_category_encoded']
    
    # Drop constant columns
    X = X.loc[:, X.std() > 0]
    
    print("Sampling data for tuning (using 100k samples for speed)...")
    # Using stratify to keep class proportions
    # 100k samples is enough for finding good params
    X_sample, _, y_sample, _ = train_test_split(X, y, train_size=100000, stratify=y, random_state=42)
    
    # Random Forest Tuning
    print("\n--- Tuning Random Forest ---")
    rf_param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf_search = RandomizedSearchCV(estimator=rf, param_distributions=rf_param_grid, 
                                   n_iter=10, scoring='f1_weighted', cv=3, random_state=42, n_jobs=1, verbose=2)
    
    start_time = time.time()
    rf_search.fit(X_sample, y_sample)
    print(f"RF Tuning completed in {time.time() - start_time:.2f}s")
    print("Best Random Forest parameters:")
    print(rf_search.best_params_)
    print(f"Best RF CV F1-Score: {rf_search.best_score_:.4f}")
    
    # XGBoost Tuning
    print("\n--- Tuning XGBoost ---")
    xgb_param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 6, 9, 12],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    xgb = XGBClassifier(random_state=42, n_jobs=-1)
    xgb_search = RandomizedSearchCV(estimator=xgb, param_distributions=xgb_param_grid, 
                                    n_iter=10, scoring='f1_weighted', cv=3, random_state=42, n_jobs=1, verbose=2)
    
    start_time = time.time()
    xgb_search.fit(X_sample, y_sample)
    print(f"XGBoost Tuning completed in {time.time() - start_time:.2f}s")
    print("Best XGBoost parameters:")
    print(xgb_search.best_params_)
    print(f"Best XGB CV F1-Score: {xgb_search.best_score_:.4f}")

if __name__ == "__main__":
    DATA = "data/processed/cleaned_earthquakes.csv"
    tune_models(DATA)
