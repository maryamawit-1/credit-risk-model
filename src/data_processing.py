import os
import joblib
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

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
# TASK 4: RFM + PROXY TARGET
# -----------------------------
def create_rfm(df):
    df = df.copy()

    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"]
    )

    snapshot_date = (
        df["TransactionStartTime"].max()
        + pd.Timedelta(days=1)
    )

    rfm = df.groupby("CustomerId").agg(
        Recency=(
            "TransactionStartTime",
            lambda x: (snapshot_date - x.max()).days
        ),
        Frequency=("TransactionStartTime", "count"),
        Monetary=("Amount", "sum")
    ).reset_index()

    return rfm


def assign_risk_clusters(rfm, n_clusters=3):
    rfm_features = rfm[
        ["Recency", "Frequency", "Monetary"]
    ].copy()

    rfm_features["Monetary"] = (
        rfm_features["Monetary"].abs()
    )

    scaler = StandardScaler()

    rfm_scaled = scaler.fit_transform(
        rfm_features
    )

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    rfm["Cluster"] = kmeans.fit_predict(
        rfm_scaled
    )

    cluster_summary = rfm.groupby(
        "Cluster"
    )[
        ["Recency", "Frequency", "Monetary"]
    ].mean()

    cluster_summary["risk_score"] = (
        cluster_summary["Recency"]
        - cluster_summary["Frequency"]
        - cluster_summary["Monetary"]
    )

    high_risk_cluster = (
        cluster_summary["risk_score"].idxmax()
    )

    rfm["is_high_risk"] = (
        rfm["Cluster"] == high_risk_cluster
    ).astype(int)

    return rfm[["CustomerId", "is_high_risk"]]

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

    # df = pd.read_csv("data/raw/data.csv")

    # X = df.drop("FraudResult", axis=1)
    # y = df["FraudResult"]

    # pipeline = build_pipeline()

    # X_processed = pipeline.fit_transform(X, y)

    df = pd.read_csv("data/raw/data.csv")

    # -----------------------------
    # TASK 4: CREATE PROXY TARGET
    # -----------------------------
    rfm = create_rfm(df)

    rfm_labels = assign_risk_clusters(rfm)

    df_with_target = df.merge(
        rfm_labels,
        on="CustomerId",
        how="left"
    )

    df_with_target["is_high_risk"] = (
        df_with_target["is_high_risk"]
        .fillna(0)
        .astype(int)
    )

    # -----------------------------
    # MODEL FEATURES + TARGET
    # -----------------------------
    X = df_with_target.drop(
        ["FraudResult", "is_high_risk"],
        axis=1
    )

    y = df_with_target["is_high_risk"]

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

    X_processed_df["is_high_risk"] = y.values

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

    print(
        "Task 4 Feature Engineering and Proxy Target Creation completed successfully."
    )