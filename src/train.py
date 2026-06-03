import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "..", "mlflow.db")

mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")
mlflow.set_experiment("credit_risk_model_task5")

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression



# -------------------------
# 1. LOAD DATA
# -------------------------
DATA_PATH = "data/processed/processed_data.csv"

df = pd.read_csv(DATA_PATH)

print("Data loaded successfully!")
print("Shape:", df.shape)
print("\nColumns:\n", df.columns)

# -------------------------
# 2. DEFINE TARGET (y)
# -------------------------
TARGET = "is_high_risk"

y = df[TARGET]

# -------------------------
# 3. DEFINE FEATURES (X)
# -------------------------
# Drop target column
X = df.drop(columns=[TARGET])

# OPTIONAL: drop obvious ID columns if they exist
possible_id_cols = ["CustomerId", "TransactionId", "BatchId"]

for col in possible_id_cols:
    if col in X.columns:
        X = X.drop(columns=[col])

print("\nFeatures shape:", X.shape)
print("Target distribution:\n", y.value_counts())

# -------------------------
# 4. TRAIN / TEST SPLIT
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nSplit completed!")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# =========================
# PART 2: BASELINE MODEL
# Logistic Regression
# =========================

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)


with mlflow.start_run(run_name="Logistic_Regression"):

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Log parameters
    mlflow.log_param("model", "LogisticRegression")

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    # Log model
    mlflow.sklearn.log_model(model, "logistic_model")

    print("\nLogistic Regression logged to MLflow")




with mlflow.start_run(run_name="Random_Forest"):

    rf_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    rf_model.fit(X_train, y_train)

    rf_pred = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]

    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_precision = precision_score(y_test, rf_pred)
    rf_recall = recall_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred)
    rf_auc = roc_auc_score(y_test, rf_proba)

    # Log parameters
    mlflow.log_param("model", "RandomForest")

    # Log metrics
    mlflow.log_metric("accuracy", rf_accuracy)
    mlflow.log_metric("precision", rf_precision)
    mlflow.log_metric("recall", rf_recall)
    mlflow.log_metric("f1", rf_f1)
    mlflow.log_metric("roc_auc", rf_auc)

    # Log model
    mlflow.sklearn.log_model(rf_model, "random_forest_model")

    print("\nRandom Forest logged to MLflow")


    # =========================
# HYPERPARAMETER TUNING (RANDOM FOREST)
# =========================

param_dist = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "class_weight": ["balanced"]
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1)

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=10,
    scoring="roc_auc",
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

best_rf = random_search.best_estimator_

print("\n=========================")
print("BEST RF PARAMETERS")
print("=========================")
print(random_search.best_params_)

best_pred = best_rf.predict(X_test)
best_proba = best_rf.predict_proba(X_test)[:, 1]

best_accuracy = accuracy_score(y_test, best_pred)
best_precision = precision_score(y_test, best_pred)
best_recall = recall_score(y_test, best_pred)
best_f1 = f1_score(y_test, best_pred)
best_auc = roc_auc_score(y_test, best_proba)

print("\n=========================")
print("TUNED RANDOM FOREST RESULTS")
print("=========================")

print("Accuracy:", best_accuracy)
print("Precision:", best_precision)
print("Recall:", best_recall)
print("F1 Score:", best_f1)
print("ROC-AUC:", best_auc)



with mlflow.start_run(run_name="Best_Tuned_Random_Forest"):

    mlflow.log_param("model", "RandomForest_Tuned")
    mlflow.log_params(random_search.best_params_)

    mlflow.log_metric("accuracy", best_accuracy)
    mlflow.log_metric("precision", best_precision)
    mlflow.log_metric("recall", best_recall)
    mlflow.log_metric("f1", best_f1)
    mlflow.log_metric("roc_auc", best_auc)

    # Log model with registered name
    model_info = mlflow.sklearn.log_model(
        sk_model=best_rf,
        artifact_path="best_random_forest_model",
        registered_model_name="CreditRiskModel"
    )

print("\nBEST MODEL LOGGED AND REGISTERED IN MLFLOW")