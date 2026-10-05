import pandas as pd
import pytest
from src.taller_retos import (
    count_customers_by_region,
    get_costa_customers_by_income,
    get_high_risk_customers,
    reconstruct_dataset_from_chunks,
)


def test_count_customers_by_region():
    df = count_customers_by_region()
    assert isinstance(df, pd.DataFrame)
    assert "region" in df.columns
    assert "total_clientes" in df.columns
    assert len(df) == 4
    # Costa tiene 4 clientes
    costa_row = df[df["region"] == "Costa"]
    assert costa_row["total_clientes"].values[0] == 4


def test_get_high_risk_customers():
    df = get_high_risk_customers(threshold=650)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert (df["credit_score"] < 650).all()


def test_get_costa_customers_by_income():
    df = get_costa_customers_by_income()
    assert isinstance(df, pd.DataFrame)
    assert (df["region"] == "Costa").all()
    assert len(df) == 4
    # Validar orden descendente
    incomes = df["annual_income"].tolist()
    assert incomes == sorted(incomes, reverse=True)


def test_reconstruct_dataset_from_chunks():
    df = reconstruct_dataset_from_chunks(chunk_size=3)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 9
    assert df["transaction_id"].is_monotonic_increasing
