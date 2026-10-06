"""Pruebas unitarias para la implementación de la spec básica en cleaning_bermudez.py."""

import numpy as np
import pandas as pd
import pytest
from scripts.cleaning_bermudez import (
    calculate_audit_metrics,
    cap_outliers,
    impute_column,
    impute_missing_values,
    normalize_frame,
    run_cleaning_pipeline,
    treat_outliers,
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


def test_normalize_frame_region_and_age_rules():
    """Spec: strip + Title Case en region; age fuera de [18, 100] -> NaN."""
    df = pd.DataFrame(
        {
            "region": ["  costa  ", "SIERRA", "oriente ", np.nan, "insular"],
            "age": [17, 18, 50, 100, 101],
        }
    )
    clean = normalize_frame(df)
    assert clean["region"].iloc[0] == "Costa"
    assert clean["region"].iloc[1] == "Sierra"
    assert clean["region"].iloc[2] == "Oriente"
    assert pd.isna(clean["region"].iloc[3])
    assert clean["region"].iloc[4] == "Insular"

    assert np.isnan(clean["age"].iloc[0])  # 17 -> NaN
    assert clean["age"].iloc[1] == 18
    assert clean["age"].iloc[2] == 50
    assert clean["age"].iloc[3] == 100
    assert np.isnan(clean["age"].iloc[4])  # 101 -> NaN


def test_impute_column_strategies():
    """Spec: mean, median, mode (categóricas), knn (KNNImputer)."""
    df = pd.DataFrame(
        {
            "val": [10.0, 20.0, 30.0, np.nan],
            "cat": ["Costa", "Costa", "Sierra", np.nan],
            "income": [10000.0, 20000.0, 80000.0, np.nan],
            "loan": [1000.0, 2000.0, 8000.0, 7500.0],
        }
    )
    # mean
    assert impute_column(df, "val", strategy="mean")["val"].iloc[3] == 20.0
    # median
    assert impute_column(df, "val", strategy="median")["val"].iloc[3] == 20.0
    # mode
    assert impute_column(df, "cat", strategy="mode")["cat"].iloc[3] == "Costa"
    # knn
    knn_res = impute_column(df, "income", strategy="knn")["income"]
    assert not knn_res.isna().any()
    assert knn_res.iloc[3] > 0.0


def test_treat_outliers_iqr_and_zscore():
    """Spec: Detección zscore (3σ) e iqr (1.5·IQR). Acción cap."""
    incomes = [30000.0, 32000.0, 35000.0, 36000.0, 38000.0, 40000.0, 42000.0, 1000000.0]
    df = pd.DataFrame({"annual_income": incomes})

    # IQR cap
    df_iqr = treat_outliers(df, "annual_income", method="iqr", action="cap")
    assert len(df_iqr) == len(df)
    assert df_iqr["annual_income"].max() < 1000000.0

    # 3-sigma cap en muestra suficiente
    data_3s = [10.0] * 30 + [500.0]
    df_3s = pd.DataFrame({"val": data_3s})
    df_3s_capped = treat_outliers(df_3s, "val", method="zscore", action="cap")
    assert len(df_3s_capped) == len(df_3s)
    assert df_3s_capped["val"].max() < 500.0


def test_run_cleaning_pipeline_bermudez_in_memory(sample_raw_dataframe):
    """Spec: run_cleaning_pipeline() orquesta todo y devuelve dict."""
    result = run_cleaning_pipeline(
        df=sample_raw_dataframe,
        save_to_db=False,
        imputation_strategy="median",
        outlier_method="iqr",
        verbose=False,
    )
    assert isinstance(result, dict)
    assert result["rows_in"] == 7
    assert result["rows_out"] == 7
    clean_df = result["dataframe"]
    assert clean_df.isna().sum().sum() == 0
    assert (clean_df["age"] >= 18).all()
    assert (clean_df["age"] <= 100).all()
    assert clean_df["annual_income"].max() < 850000.0
