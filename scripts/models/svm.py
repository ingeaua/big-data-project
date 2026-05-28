import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import joblib
import os

# ── Constants ──────────────────────────────────────────────────────────────────
PROCESSED_DATA = 'earthquake_project/data/processed/cleaned_earthquakes.csv'
EXCLUDE_COLS = ['mag', 'mag_category', 'mag_category_encoded']
TEST_SIZE = 0.2
RANDOM_STATE = 42
MODELS_DIR = 'earthquake_project/results/models'

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(PROCESSED_DATA)

X = df.drop(columns=EXCLUDE_COLS)
y = df['mag_category_encoded']

# Drop constant columns
constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
if constant_cols:
    print(f"Dropping constant columns: {constant_cols}")
    X = X.drop(columns=constant_cols)

print(f"Features shape: {X.shape}")

# ── Train/test split ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# ── Feature scaling ───────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ── Train model ───────────────────────────────────────────────────────────────
print("Training LinearSVC...")
model = LinearSVC(max_iter=2000, random_state=RANDOM_STATE, dual='auto')
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM Accuracy: {accuracy:.4f}")

# ── Save model ────────────────────────────────────────────────────────────────
os.makedirs(MODELS_DIR, exist_ok=True)
joblib.dump({'model': model, 'scaler': scaler}, os.path.join(MODELS_DIR, 'svm.joblib'))
print(f"Model saved to {MODELS_DIR}/svm.joblib")
