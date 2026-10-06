"""Pruebas unitarias para la implementación de la spec básica en cleaning_bermudez.py."""

import numpy as np
import pandas as pd
import pytest
from scripts.cleaning_bermudez import (
    calculate_audit_metrics,
    cap_outliers,
    clean_inconsistencies,
    impute_column,
    impute_missing_values,
    normalize_inconsistencies,
    run_cleaning_pipeline,
)
from src.db_connector import get_database_engine


@pytest.fixture
def sample_raw_dataframe():
    """Genera un DataFrame sintético con la Tríada Patológica de Datos."""
    return pd.DataFrame(
        {
            "transaction_id": [1, 2, 3, 4, 5, 6, 7],
            "customer_id": ["C1", "C2", "C3", "C4", "C5", "C6", "C7"],
            "age": [25, 12, 105, 40, np.nan, 30, 55],
            "annual_income": [30000.0, 45000.0, 50000.0, 42000.0, np.nan, 48000.0, 850000.0],
            "credit_score": [700, 650, 720, 680, 710, np.nan, 750],
            "loan_amount": [5000.0, 10000.0, 8000.0, 12000.0, 7000.0, 9000.0, 300000.0],
            "region": ["  costa  ", "SIERRA", "oriente", "costa", np.nan, "insular ", "Costa"],
        }
    )


def test_clean_inconsistencies_region():
    df = pd.DataFrame({"region": ["  costa  ", "SIERRA", "oriente ", np.nan, "insular"]})
    df_clean = clean_inconsistencies(df)
    assert df_clean["region"].iloc[0] == "Costa"
    assert df_clean["region"].iloc[1] == "Sierra"
    assert df_clean["region"].iloc[2] == "Oriente"
    assert pd.isna(df_clean["region"].iloc[3])
    assert df_clean["region"].iloc[4] == "Insular"


def test_clean_inconsistencies_age_rules():
    df = pd.DataFrame({"age": [17, 18, 50, 100, 101, -5, np.nan]})
    df_clean = clean_inconsistencies(df)
    assert np.isnan(df_clean["age"].iloc[0])
    assert df_clean["age"].iloc[1] == 18
    assert df_clean["age"].iloc[2] == 50
    assert df_clean["age"].iloc[3] == 100
    assert np.isnan(df_clean["age"].iloc[4])
    assert np.isnan(df_clean["age"].iloc[5])
    assert np.isnan(df_clean["age"].iloc[6])


def test_impute_column_strategies():
    df = pd.DataFrame({"val": [10.0, 20.0, 30.0, np.nan], "cat": ["A", "A", "B", np.nan]})
    assert impute_column(df, "val", strategy="mean")["val"].iloc[3] == 20.0
    assert impute_column(df, "val", strategy="median")["val"].iloc[3] == 20.0
    assert impute_column(df, "cat", strategy="mode")["cat"].iloc[3] == "A"


def test_cap_outliers_iqr():
    data = [30000.0, 32000.0, 35000.0, 36000.0, 38000.0, 40000.0, 42000.0, 1000000.0]
    df = pd.DataFrame({"annual_income": data})
    df_capped = cap_outliers(df, columns=["annual_income"], method="iqr")
    assert len(df_capped) == len(df)
    assert df_capped["annual_income"].max() < 1000000.0
    assert df_capped["annual_income"].var() < df["annual_income"].var()


def test_run_cleaning_pipeline_bermudez_in_memory(sample_raw_dataframe):
    df_clean = run_cleaning_pipeline(
        df=sample_raw_dataframe,
        save_to_db=False,
        imputation_strategy="median",
        outlier_method="iqr",
        verbose=False,
    )
    assert df_clean.isna().sum().sum() == 0
    assert len(df_clean) == len(sample_raw_dataframe)
    assert (df_clean["age"] >= 18).all()
    assert (df_clean["age"] <= 100).all()
    assert df_clean["annual_income"].max() < 850000.0
