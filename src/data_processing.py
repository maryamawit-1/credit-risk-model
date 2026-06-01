import os
import joblib
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from category_encoders.woe import WOEEncoder


# -----------------------------
# Custom Feature Engineering
# -----------------------------
class AggregateFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        customer_agg = X.groupby('CustomerId')['Amount'].agg([
            'sum',
            'mean',
            'count',
            'std'
        ]).reset_index()

        customer_agg.columns = [
            'CustomerId',
            'total_transaction_amount',
            'avg_transaction_amount',
            'transaction_count',
            'std_transaction_amount'
        ]

        X = X.merge(customer_agg, on='CustomerId', how='left')

        return X


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])

        X['transaction_hour'] = X['TransactionStartTime'].dt.hour
        X['transaction_day'] = X['TransactionStartTime'].dt.day
        X['transaction_month'] = X['TransactionStartTime'].dt.month
        X['transaction_year'] = X['TransactionStartTime'].dt.year

        return X


class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns, errors='ignore')


# -----------------------------
# Build Pipeline
# -----------------------------
def build_pipeline():

    numeric_features = [
        'Amount',
        'CountryCode',
        'total_transaction_amount',
        'avg_transaction_amount',
        'transaction_count',
        'std_transaction_amount',
        'transaction_hour',
        'transaction_day',
        'transaction_month',
        'transaction_year'
    ]

    categorical_features = [
        'CurrencyCode',
        'ProviderId',
        'ProductId',
        'ProductCategory',
        'ChannelId',
        'PricingStrategy'
    ]

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    pipeline = Pipeline([
        ('aggregate_features', AggregateFeatures()),
        ('date_features', DateFeatureExtractor()),
        ('drop_columns', DropColumns([
            'TransactionId',
            'BatchId',
            'AccountId',
            'SubscriptionId',
            'TransactionStartTime',
            'Value'
        ])),
        ('preprocessor', preprocessor)
    ])

    return pipeline


# -----------------------------
# Run Processing
# -----------------------------
if __name__ == "__main__":

    df = pd.read_csv("data/raw/data.csv")

    X = df.drop('FraudResult', axis=1)
    y = df['FraudResult']

    pipeline = build_pipeline()

    X_processed = pipeline.fit_transform(X, y)

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # pd.DataFrame(X_processed.toarray()).to_csv(
    #     "data/processed/processed_data.csv",
    #     index=False
    # )

    # Get feature names from pipeline
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()

    # Convert to DataFrame with headers
    X_processed_df = pd.DataFrame(
        X_processed.toarray(),
        columns=feature_names
    )

    # Save
    X_processed_df.to_csv(
        "data/processed/processed_data.csv",
        index=False
    )

    joblib.dump(
        pipeline,
        "models/data_processing_pipeline.pkl"
    )

    print("Data processing completed successfully.")
