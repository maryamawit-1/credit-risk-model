import os
import joblib
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from category_encoders.woe import WOEEncoder


# -----------------------------
# Feature Engineering
# -----------------------------
class AggregateFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        agg = X.groupby("CustomerId")["Amount"].agg(
            total_transaction_amount="sum",
            avg_transaction_amount="mean",
            transaction_count="count",
            std_transaction_amount="std"
        ).reset_index()

        return X.merge(agg, on="CustomerId", how="left")


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X["TransactionStartTime"] = pd.to_datetime(X["TransactionStartTime"])

        X["transaction_hour"] = X["TransactionStartTime"].dt.hour
        X["transaction_day"] = X["TransactionStartTime"].dt.day
        X["transaction_month"] = X["TransactionStartTime"].dt.month
        X["transaction_year"] = X["TransactionStartTime"].dt.year

        return X


class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns, errors="ignore")


# -----------------------------
# IV CALCULATION
# -----------------------------
def calculate_iv(df, feature, target):
    iv_data = []

    for val in df[feature].unique():
        good = len(df[(df[feature] == val) & (df[target] == 0)])
        bad = len(df[(df[feature] == val) & (df[target] == 1)])

        iv_data.append([val, good, bad])

    iv_df = pd.DataFrame(iv_data, columns=["value", "good", "bad"])

    iv_df["good_dist"] = iv_df["good"] / (iv_df["good"].sum() + 1e-6)
    iv_df["bad_dist"] = iv_df["bad"] / (iv_df["bad"].sum() + 1e-6)

    iv_df["woe"] = np.log((iv_df["good_dist"] + 1e-6) / (iv_df["bad_dist"] + 1e-6))

    iv_df["iv_component"] = (iv_df["good_dist"] - iv_df["bad_dist"]) * iv_df["woe"]

    return iv_df["iv_component"].sum()


# -----------------------------
# Pipeline
# -----------------------------
def build_pipeline():

    numeric_features = [
        "Amount",
        "CountryCode",
        "total_transaction_amount",
        "avg_transaction_amount",
        "transaction_count",
        "std_transaction_amount",
        "transaction_hour",
        "transaction_day",
        "transaction_month",
        "transaction_year"
    ]

    categorical_features = [
        "CurrencyCode",
        "ProviderId",
        "ProductId",
        "ProductCategory",
        "ChannelId",
        "PricingStrategy"
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("woe", WOEEncoder())
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    pipeline = Pipeline([
        ("aggregate", AggregateFeatures()),
        ("date_features", DateFeatureExtractor()),
        ("drop", DropColumns([
            "TransactionId",
            "BatchId",
            "AccountId",
            "SubscriptionId",
            "TransactionStartTime",
            "Value"
        ])),
        ("preprocessor", preprocessor)
    ])

    return pipeline


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":

    df = pd.read_csv("data/raw/data.csv")

    X = df.drop("FraudResult", axis=1)
    y = df["FraudResult"]

    pipeline = build_pipeline()

    X_processed = pipeline.fit_transform(X, y)

    # -----------------------------
    # Feature names
    # -----------------------------
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

    X_processed_df = pd.DataFrame(
        X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed,
        columns=feature_names
    )

    # -----------------------------
    # Save processed data + model
    # -----------------------------
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    X_processed_df.to_csv("data/processed/processed_data.csv", index=False)

    joblib.dump(pipeline, "models/data_processing_pipeline.pkl")

    # -----------------------------
    # IV CALCULATION
    # -----------------------------
    categorical_features = [
        "CurrencyCode",
        "ProviderId",
        "ProductId",
        "ProductCategory",
        "ChannelId",
        "PricingStrategy"
    ]

    iv_results = []

    for col in categorical_features:
        iv = calculate_iv(df, col, "FraudResult")
        iv_results.append([col, iv])

    iv_df = pd.DataFrame(iv_results, columns=["feature", "iv"])

    iv_df.to_csv("data/processed/iv_values.csv", index=False)

    print("Task 3 Feature Engineering completed successfully.")