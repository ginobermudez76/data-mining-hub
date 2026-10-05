"""Suite de pruebas unitarias y de integración para el módulo de limpieza y calidad de datos."""

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text
from src.cleaning import (
    calculate_audit_metrics,
    cap_outliers,
    clean_inconsistencies,
    impute_column,
    impute_missing_values,
    normalize_inconsistencies,
    run_cleaning_pipeline,
)
from src.db_connector import extract_raw_data, get_database_engine


@pytest.fixture
def sample_raw_dataframe():
    """Genera un DataFrame sintético con la Tríada Patológica de Datos."""
    return pd.DataFrame(
        {
            "transaction_id": [1, 2, 3, 4, 5, 6, 7],
            "customer_id": ["C1", "C2", "C3", "C4", "C5", "C6", "C7"],
            "age": [25, 12, 105, 40, np.nan, 30, 55],  # 12 y 105 violan reglas (<18, >100)
            "annual_income": [30000.0, 45000.0, 50000.0, 42000.0, np.nan, 48000.0, 850000.0],  # 850000 es outlier
            "credit_score": [700, 650, 720, 680, 710, np.nan, 750],
            "loan_amount": [5000.0, 10000.0, 8000.0, 12000.0, 7000.0, 9000.0, 300000.0],  # 300000 es outlier
            "region": ["  costa  ", "SIERRA", "oriente", "costa", np.nan, "insular ", "Costa"],
        }
    )


# ==============================================================================
# ACTIVIDAD 3.1: GESTIÓN DE INCONSISTENCIAS
# ==============================================================================

def test_clean_inconsistencies_region():
    """Valida strip y Title Case en la columna region, preservando nulos."""
    df = pd.DataFrame({"region": ["  costa  ", "SIERRA", "oriente ", np.nan, "insular"]})
    df_clean = clean_inconsistencies(df)

    assert df_clean["region"].iloc[0] == "Costa"
    assert df_clean["region"].iloc[1] == "Sierra"
    assert df_clean["region"].iloc[2] == "Oriente"
    assert pd.isna(df_clean["region"].iloc[3])
    assert df_clean["region"].iloc[4] == "Insular"


def test_clean_inconsistencies_age_rules():
    """Valida que edades < 18 o > 100 se transformen estructuralmente en NaN."""
    df = pd.DataFrame({"age": [17, 18, 50, 100, 101, -5, np.nan]})
    df_clean = clean_inconsistencies(df)

    # 17, 101, -5 deben ser NaN
    assert np.isnan(df_clean["age"].iloc[0])
    assert df_clean["age"].iloc[1] == 18
    assert df_clean["age"].iloc[2] == 50
    assert df_clean["age"].iloc[3] == 100
    assert np.isnan(df_clean["age"].iloc[4])
    assert np.isnan(df_clean["age"].iloc[5])
    assert np.isnan(df_clean["age"].iloc[6])


def test_normalize_inconsistencies_alias():
    """Valida que el alias normalize_inconsistencies funcione de forma idéntica."""
    df = pd.DataFrame({"region": ["  costa "], "age": [15]})
    df_clean = normalize_inconsistencies(df)

    assert df_clean["region"].iloc[0] == "Costa"
    assert np.isnan(df_clean["age"].iloc[0])


# ==============================================================================
# ACTIVIDAD 3.2: TRATAMIENTO DE VALORES FALTANTES (IMPUTACIÓN)
# ==============================================================================

def test_impute_column_mean():
    """Valida la imputación estadística por media matemática."""
    df = pd.DataFrame({"salary": [10.0, 20.0, 30.0, np.nan]})
    # Media de 10, 20, 30 es 20.0
    df_imp = impute_column(df, "salary", strategy="mean")

    assert not df_imp["salary"].isna().any()
    assert df_imp["salary"].iloc[3] == 20.0


def test_impute_column_median():
    """Valida la imputación estadística por mediana."""
    df = pd.DataFrame({"val": [10.0, 20.0, 1000.0, np.nan]})
    # Mediana de 10, 20, 1000 es 20.0 (resistente al outlier 1000)
    df_imp = impute_column(df, "val", strategy="median")

    assert not df_imp["val"].isna().any()
    assert df_imp["val"].iloc[3] == 20.0


def test_impute_column_mode():
    """Valida la imputación por moda en variables categóricas."""
    df = pd.DataFrame({"region": ["Costa", "Sierra", "Costa", np.nan, "Costa"]})
    df_imp = impute_column(df, "region", strategy="mode")

    assert not df_imp["region"].isna().any()
    assert df_imp["region"].iloc[3] == "Costa"


def test_impute_column_knn():
    """Valida la imputación multivariable con KNNImputer."""
    df = pd.DataFrame(
        {
            "income": [10000.0, 20000.0, 80000.0, 90000.0, np.nan],
            "loan": [1000.0, 2000.0, 8000.0, 9000.0, 8500.0],
        }
    )
    # El valor nulo con loan=8500 debe imputarse cercano al cluster alto (80000-90000)
    df_imp = impute_column(df, "income", strategy="knn", n_neighbors=2)

    assert not df_imp["income"].isna().any()
    assert df_imp["income"].iloc[4] > 50000.0


def test_impute_column_validations():
    """Valida manejo de errores y validaciones en impute_column."""
    df = pd.DataFrame({"text_col": ["a", "b", np.nan], "num_col": [1.0, 2.0, np.nan]})

    # Columna no existente
    with pytest.raises(KeyError):
        impute_column(df, "columna_fantasma", strategy="mean")

    # Estrategia no válida
    with pytest.raises(ValueError):
        impute_column(df, "num_col", strategy="estrategia_desconocida")

    # Estrategia numérica sobre columna de texto
    with pytest.raises(ValueError):
        impute_column(df, "text_col", strategy="mean")


def test_impute_missing_values_pipeline():
    """Valida la imputación automática completa de todas las columnas con nulos."""
    df = pd.DataFrame(
        {
            "age": [20.0, 30.0, np.nan],
            "income": [1000.0, np.nan, 3000.0],
            "region": ["Costa", "Sierra", np.nan],
        }
    )
    df_imp = impute_missing_values(df, numeric_strategy="median", categorical_strategy="mode")

    assert df_imp.isna().sum().sum() == 0
    assert df_imp["age"].iloc[2] == 25.0
    assert df_imp["income"].iloc[1] == 2000.0
    assert df_imp["region"].iloc[2] in ["Costa", "Sierra"]


# ==============================================================================
# ACTIVIDAD 3.3: MANEJO ALGORÍTMICO DE OUTLIERS (CAPPING)
# ==============================================================================

def test_cap_outliers_iqr():
    """Valida la detección y acotamiento por IQR sin eliminar registros."""
    # Conjunto con valores normales y un outlier extremo (1,000,000)
    incomes = [30000.0, 32000.0, 35000.0, 36000.0, 38000.0, 40000.0, 42000.0, 1000000.0]
    df = pd.DataFrame({"annual_income": incomes})
    initial_len = len(df)

    df_capped = cap_outliers(df, columns=["annual_income"], method="iqr", factor=1.5)

    # 1. Se debe preservar el 100% de los registros
    assert len(df_capped) == initial_len

    # 2. El valor extremo debe haber sido acotado (menor a 1,000,000)
    max_capped = df_capped["annual_income"].max()
    assert max_capped < 1000000.0

    # 3. La varianza debe ser significativamente menor
    assert df_capped["annual_income"].var() < df["annual_income"].var()


def test_cap_outliers_3sigma():
    """Valida la detección y acotamiento por regla empírica 3-sigma."""
    data = [10.0] * 30 + [500.0]
    df = pd.DataFrame({"annual_income": data})
    df_capped = cap_outliers(df, columns=["annual_income"], method="3sigma")

    assert len(df_capped) == len(df)
    assert df_capped["annual_income"].max() < 500.0
    assert df_capped["annual_income"].var() < df["annual_income"].var()


# ==============================================================================
# ACTIVIDAD 4: PIPELINE FINAL Y PERSISTENCIA
# ==============================================================================

def test_run_cleaning_pipeline_in_memory(sample_raw_dataframe):
    """Valida la ejecución secuencial completa del pipeline en memoria."""
    df_clean = run_cleaning_pipeline(
        df=sample_raw_dataframe,
        save_to_db=False,
        imputation_strategy="median",
        outlier_method="iqr",
        verbose=False,
    )

    # Validar que no hay valores nulos residuales
    assert df_clean.isna().sum().sum() == 0

    # Validar que se preservan todas las filas (0 eliminaciones)
    assert len(df_clean) == len(sample_raw_dataframe)

    # Validar que region está en Title Case sin espacios
    assert (df_clean["region"] == df_clean["region"].str.strip().str.title()).all()

    # Validar que todas las edades cumplen regla de negocio
    assert (df_clean["age"] >= 18).all()
    assert (df_clean["age"] <= 100).all()

    # Validar que los outliers fueron acotados
    assert df_clean["annual_income"].max() < 850000.0
    assert df_clean["loan_amount"].max() < 300000.0


def test_run_cleaning_pipeline_database_integration():
    """Valida que el pipeline extraiga de PostgreSQL y guarde en customer_credit_clean."""
    engine = get_database_engine()

    df_clean = run_cleaning_pipeline(
        engine=engine,
        save_to_db=True,
        target_table="customer_credit_clean",
        imputation_strategy="median",
        outlier_method="iqr",
        verbose=False,
    )

    # Consultar la tabla guardada directamente con SQL
    with engine.connect() as conn:
        df_db = pd.read_sql_table("customer_credit_clean", con=conn)

    assert not df_db.empty
    assert len(df_db) == len(df_clean)
    assert df_db.isna().sum().sum() == 0
    assert "region" in df_db.columns
    assert "annual_income" in df_db.columns


def test_calculate_audit_metrics():
    """Valida el cálculo de métricas de auditoría y comparación de varianza."""
    df_before = pd.DataFrame({"annual_income": [10.0, 20.0, 1000.0, np.nan]})
    df_after = pd.DataFrame({"annual_income": [10.0, 20.0, 50.0, 25.0]})

    metrics = calculate_audit_metrics(df_before, df_after, column="annual_income")

    assert metrics["records_before"] == 4
    assert metrics["records_after"] == 4
    assert metrics["nulls_before"] == 1
    assert metrics["nulls_after"] == 0
    assert metrics["variance_after"] < metrics["variance_before"]
    assert metrics["variance_difference"] < 0
