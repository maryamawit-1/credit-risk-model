import os
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# -------------------------
# MLflow setup
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "..", "mlflow.db")

mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")
mlflow.set_experiment("credit_risk_model_task5")

# -------------------------
# Load data
# -------------------------
df = pd.read_csv("data/processed/processed_data.csv")

TARGET = "is_high_risk"

y = df[TARGET]
X = df.drop(columns=[TARGET])

# remove IDs if present
for col in ["CustomerId", "TransactionId", "BatchId"]:
    if col in X.columns:
        X = X.drop(columns=[col])

# -------------------------
# split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 1. Logistic Regression
# =========================
with mlflow.start_run(run_name="Logistic_Regression"):

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)

    pred = lr.predict(X_test)
    proba = lr.predict_proba(X_test)[:, 1]

    mlflow.log_metric("roc_auc", roc_auc_score(y_test, proba))
    mlflow.log_metric("f1", f1_score(y_test, pred))

    mlflow.sklearn.log_model(lr, "logistic_model")

# Save local model (IMPORTANT FOR API)
best_local_model = lr
best_model_name = "LogisticRegression"

best_score = roc_auc_score(y_test, proba)

# =========================
# 2. Random Forest + tuning
# =========================
param_dist = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "class_weight": ["balanced"]
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1)

search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=10,
    scoring="roc_auc",
    cv=3,
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)

best_rf = search.best_estimator_

pred = best_rf.predict(X_test)
proba = best_rf.predict_proba(X_test)[:, 1]

rf_score = roc_auc_score(y_test, proba)

with mlflow.start_run(run_name="Random_Forest_Tuned"):

    mlflow.log_params(search.best_params_)
    mlflow.log_metric("roc_auc", rf_score)

    mlflow.sklearn.log_model(
        best_rf,
        "best_rf_model",
        registered_model_name="CreditRiskModel"
    )

# -------------------------
# PICK BEST MODEL
# -------------------------
if rf_score > best_score:
    final_model = best_rf
    print("Best model: Random Forest")
else:
    final_model = lr
    print("Best model: Logistic Regression")

# -------------------------
# SAVE MODEL FOR API (CRITICAL FIX)
# -------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(final_model, "models/final_model.pkl")

print("Model saved to models/final_model.pkl")