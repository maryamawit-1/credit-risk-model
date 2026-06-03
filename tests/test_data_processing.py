import pandas as pd

def test_data_load_shape():
    df = pd.read_csv("data/processed/processed_data.csv")
    assert df.shape[1] >= 5   # dataset must have features


def test_target_column_exists():
    df = pd.read_csv("data/processed/processed_data.csv")
    assert "is_high_risk" in df.columns


def test_no_missing_target():
    df = pd.read_csv("data/processed/processed_data.csv")
    assert df["is_high_risk"].isnull().sum() == 0