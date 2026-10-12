"""Pruebas unitarias para el Pipeline de Ingesta, Exploración y Calidad de Datos.

Autor: Gino Maximiliano Bermúdez Santos
Módulo: apps/analytics/tests/test_pipeline_calidad.py
"""

import sqlite3
import numpy as np
import pandas as pd
import pytest

from src.pipeline_calidad import (
    create_sqlite_database,
    extract_raw_data,
    calculate_eda_metrics,
    clean_inconsistencies,
    impute_missing_values,
    calculate_iqr_bounds,
    treat_outliers_iqr,
    compute_comparative_metrics,
    export_curated_data,
    run_full_pipeline,
    RAW_CLIENTES_DATA,
)


def test_create_and_extract_raw_data():
    """Fase 1: Conexión SQLite, ejecución estructurada y verificación de shape y dtypes."""
    conn = create_sqlite_database(in_memory=True)
    df = extract_raw_data(conn)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (15, 6)
    expected_cols = ["id_cliente", "edad", "ingreso_mensual", "ciudad", "antiguedad_meses", "nivel_riesgo"]
    assert list(df.columns) == expected_cols

    # Verificación de tipos de datos relacionales esperados
    assert np.issubdtype(df["id_cliente"].dtype, np.integer)
    assert np.issubdtype(df["edad"].dtype, np.floating)
    assert np.issubdtype(df["ingreso_mensual"].dtype, np.floating)
    assert np.issubdtype(df["antiguedad_meses"].dtype, np.floating)


def test_eda_metrics_and_skewness():
    """Fase 2: Estadísticas descriptivas y detección de asimetría positiva en ingresos."""
    conn = create_sqlite_database(in_memory=True)
    df = extract_raw_data(conn)
    eda = calculate_eda_metrics(df)

    assert eda["shape"] == (15, 6)
    assert eda["null_counts"]["ingreso_mensual"] == 2
    assert eda["null_counts"]["edad"] == 1

    # Asimetría fuertemente positiva debido al outlier de 95000
    skewness = eda["skewness_ingreso_mensual"]
    assert skewness > 3.0  # ~3.597


def test_clean_inconsistencies_categorical_and_business_rules():
    """Fase 3: Estandarización de ciudad y reemplazo por NaN de valores imposibles."""
    conn = create_sqlite_database(in_memory=True)
    df = extract_raw_data(conn)
    df_cleaned = clean_inconsistencies(df)

    # 1. Estandarización de ciudad
    ciudades_esperadas = {"Guayaquil", "Santa Elena", "La Libertad", "Salinas", "Manta"}
    assert set(df_cleaned["ciudad"].unique()) == ciudades_esperadas
    assert "guayaquil" not in df_cleaned["ciudad"].values
    assert "SANTA ELENA" not in df_cleaned["ciudad"].values
    assert "salinas" not in df_cleaned["ciudad"].values

    # 2. Reglas de negocio: edad (< 18 o > 100) -> NaN
    # Originales: id 3 (-5.0), id 12 (140.0), id 6 (ya era NaN)
    nulos_edad = df_cleaned["edad"].isnull().sum()
    assert nulos_edad == 3
    assert np.isnan(df_cleaned.loc[df_cleaned["id_cliente"] == 3, "edad"].values[0])
    assert np.isnan(df_cleaned.loc[df_cleaned["id_cliente"] == 12, "edad"].values[0])

    # 3. Reglas de negocio: antiguedad_meses (< 0) -> NaN
    # Original: id 7 (-2.0)
    assert np.isnan(df_cleaned.loc[df_cleaned["id_cliente"] == 7, "antiguedad_meses"].values[0])


def test_impute_missing_values_median():
    """Fase 3: Imputación de edad y antiguedad con la mediana calculada sobre válidos."""
    conn = create_sqlite_database(in_memory=True)
    df_raw = extract_raw_data(conn)
    df_inconsist = clean_inconsistencies(df_raw)

    df_imputed = impute_missing_values(df_inconsist, method_ingreso="conditional_risk")

    # Mediana esperada de edad = 37.0
    assert df_imputed["edad"].isnull().sum() == 0
    assert df_imputed.loc[df_imputed["id_cliente"] == 3, "edad"].values[0] == 37.0
    assert df_imputed.loc[df_imputed["id_cliente"] == 6, "edad"].values[0] == 37.0
    assert df_imputed.loc[df_imputed["id_cliente"] == 12, "edad"].values[0] == 37.0

    # Mediana esperada de antiguedad = 22.0
    assert df_imputed["antiguedad_meses"].isnull().sum() == 0
    assert df_imputed.loc[df_imputed["id_cliente"] == 7, "antiguedad_meses"].values[0] == 22.0


def test_impute_missing_values_income_conditional_and_knn():
    """Fase 3: Imputación condicional de ingresos por nivel_riesgo y por KNN."""
    conn = create_sqlite_database(in_memory=True)
    df_raw = extract_raw_data(conn)
    df_inconsist = clean_inconsistencies(df_raw)

    # 1. Estrategia condicional por nivel_riesgo
    df_cond = impute_missing_values(df_inconsist, method_ingreso="conditional_risk")
    assert df_cond["ingreso_mensual"].isnull().sum() == 0
    # id 4 es 'Alto' -> mediana de Alto es 4200.0
    assert df_cond.loc[df_cond["id_cliente"] == 4, "ingreso_mensual"].values[0] == 4200.0
    # id 9 es 'Bajo' -> mediana de Bajo es 1500.0
    assert df_cond.loc[df_cond["id_cliente"] == 9, "ingreso_mensual"].values[0] == 1500.0

    # 2. Estrategia KNN
    df_knn = impute_missing_values(df_inconsist, method_ingreso="knn", k_neighbors=3)
    assert df_knn["ingreso_mensual"].isnull().sum() == 0
    assert not np.isnan(df_knn.loc[df_knn["id_cliente"] == 4, "ingreso_mensual"].values[0])


def test_iqr_bounds_and_outlier_treatment():
    """Fase 3: Detección matemática de outliers por IQR y tratamiento (Winsorización vs Eliminación)."""
    conn = create_sqlite_database(in_memory=True)
    df_raw = extract_raw_data(conn)
    df_step1 = clean_inconsistencies(df_raw)
    df_step2 = impute_missing_values(df_step1, method_ingreso="conditional_risk")

    # 1. Prueba de límites IQR
    bounds = calculate_iqr_bounds(df_step2["ingreso_mensual"], multiplier=1.5)
    assert bounds["q1"] == 1550.0
    assert bounds["q3"] == 3300.0
    assert bounds["iqr"] == 1750.0
    assert bounds["limite_inferior"] == -1075.0
    assert bounds["limite_superior"] == 5925.0

    # 2. Winsorización
    df_winsor, outlier_idx, _ = treat_outliers_iqr(
        df_step2, column="ingreso_mensual", action="winsorize"
    )
    assert outlier_idx == [4]  # índice 4 -> id_cliente 5 (95000.0)
    assert len(df_winsor) == 15
    assert df_winsor.loc[4, "ingreso_mensual"] == 5925.0

    # 3. Eliminación
    df_elim, outlier_idx_elim, _ = treat_outliers_iqr(
        df_step2, column="ingreso_mensual", action="eliminate"
    )
    assert len(df_elim) == 14
    assert 5 not in df_elim["id_cliente"].values


def test_comparative_metrics_and_sqlite_export():
    """Fase 4: Comparativa de estadísticas descriptivas y exportación a clientes_curados."""
    res = run_full_pipeline(method_ingreso="conditional_risk", outlier_action="winsorize")

    comparativa = res["comparativa"]
    assert "edad" in comparativa.index
    assert "ingreso_mensual" in comparativa.index
    assert "antiguedad_meses" in comparativa.index

    # La desviación estándar de ingreso_mensual debe haber caído drásticamente
    assert comparativa.loc["ingreso_mensual", "clean_std"] < comparativa.loc["ingreso_mensual", "raw_std"]
    assert comparativa.loc["ingreso_mensual", "reduccion_std_pct"] > 90.0

    # Verificación en la base de datos relacional
    conn = res["connection"]
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM clientes_curados")
    count_curados = cur.fetchone()[0]
    assert count_curados == 15

    # Verificación de que no hay nulos en la tabla curada
    cur.execute("""
    SELECT COUNT(*) FROM clientes_curados
    WHERE edad IS NULL
       OR ingreso_mensual IS NULL
       OR ciudad IS NULL
       OR antiguedad_meses IS NULL
       OR nivel_riesgo IS NULL
    """)
    nulos = cur.fetchone()[0]
    assert nulos == 0
