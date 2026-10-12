"""Pipeline de Ingesta, Exploración y Calidad de Datos (Unidad 2 - Data Engineering).

Autor: Gino Maximiliano Bermúdez Santos
Asignatura: Minería de Datos
Módulo: apps/analytics/src/pipeline_calidad.py
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import sqlite3
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

# Definición del dataset sintético crudo según la Guía Práctica
RAW_CLIENTES_DATA: List[Tuple[int, Optional[float], Optional[float], str, Optional[float], str]] = [
    (1, 25.0, 1200.0, "Guayaquil", 12.0, "Bajo"),
    (2, 34.0, 2100.0, "guayaquil", 24.0, "Medio"),
    (3, -5.0, 1800.0, "Santa Elena", 6.0, "Bajo"),
    (4, 45.0, np.nan, "SANTA ELENA", 48.0, "Alto"),
    (5, 52.0, 95000.0, "La Libertad", 60.0, "Alto"),
    (6, np.nan, 1400.0, "Salinas", 10.0, "Bajo"),
    (7, 29.0, 1600.0, "salinas", -2.0, "Bajo"),
    (8, 41.0, 3100.0, "Guayaquil", 36.0, "Medio"),
    (9, 38.0, np.nan, "Manta", 18.0, "Bajo"),
    (10, 60.0, 3500.0, "Guayaquil", 120.0, "Alto"),
    (11, 22.0, 900.0, "Salinas", 4.0, "Bajo"),
    (12, 140.0, 2200.0, "Santa Elena", 15.0, "Medio"),
    (13, 31.0, 1950.0, "La Libertad", 20.0, "Bajo"),
    (14, 48.0, 4200.0, "guayaquil", 50.0, "Alto"),
    (15, 36.0, 2300.0, "Santa Elena", 25.0, "Medio"),
]


# ==============================================================================
# FASE 1: EXTRACCIÓN RELACIONAL Y VERIFICACIÓN INICIAL
# ==============================================================================

def create_sqlite_database(
    in_memory: bool = True,
    db_path: Optional[str] = None
) -> sqlite3.Connection:
    """Crea la base de datos relacional SQLite e inicializa la tabla clientes_crudo.

    Args:
        in_memory: Si es True, conecta a ':memory:'.
        db_path: Ruta al archivo SQLite si no es en memoria.

    Returns:
        sqlite3.Connection: Conexión activa con datos insertados.
    """
    target = ":memory:" if in_memory or db_path is None else db_path
    conn = sqlite3.connect(target)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS clientes_crudo (
        id_cliente INTEGER PRIMARY KEY,
        edad REAL,
        ingreso_mensual REAL,
        ciudad TEXT,
        antiguedad_meses REAL,
        nivel_riesgo TEXT
    );
    """)

    cursor.executemany(
        "INSERT INTO clientes_crudo VALUES (?, ?, ?, ?, ?, ?)",
        RAW_CLIENTES_DATA
    )
    conn.commit()
    return conn


def extract_raw_data(
    conn: sqlite3.Connection,
    query: str = "SELECT * FROM clientes_crudo"
) -> pd.DataFrame:
    """Extrae los datos crudos ejecutando una consulta SQL estructurada.

    Args:
        conn: Conexión SQLite activa.
        query: Sentencia SQL a ejecutar.

    Returns:
        pd.DataFrame: DataFrame de pandas con los datos crudos extraídos.
    """
    return pd.read_sql_query(query, conn)


# ==============================================================================
# FASE 2: ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ==============================================================================

def calculate_eda_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Genera métricas descriptivas completas y cálculo de asimetría.

    Args:
        df: DataFrame de entrada.

    Returns:
        Dict[str, Any]: Estadísticas descriptivas, skewness y dimensiones.
    """
    skewness_ingreso = float(df["ingreso_mensual"].skew()) if "ingreso_mensual" in df.columns else np.nan
    desc_df = df.describe(include="all")

    return {
        "shape": df.shape,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "null_counts": df.isnull().sum().to_dict(),
        "skewness_ingreso_mensual": skewness_ingreso,
        "describe": desc_df.to_dict(),
    }


# ==============================================================================
# FASE 3: TRATAMIENTO ALGORÍTMICO Y CALIDAD DE DATOS
# ==============================================================================

def clean_inconsistencies(df: pd.DataFrame) -> pd.DataFrame:
    """Corrige inconsistencias categóricas y de reglas de negocio:
    1. Estandariza columna 'ciudad' (espacios en blanco y Title Case).
    2. Convierte a NaN valores imposibles en 'antiguedad_meses' (< 0).
    3. Convierte a NaN valores imposibles en 'edad' (< 18 o > 100).

    Args:
        df: DataFrame original.

    Returns:
        pd.DataFrame: Copia del DataFrame con inconsistencias corregidas a NaN.
    """
    out = df.copy()

    # 1. Estandarización categórica de ciudad
    if "ciudad" in out.columns:
        out["ciudad"] = out["ciudad"].apply(
            lambda x: str(x).strip().title() if pd.notna(x) else x
        )

    # 2. Reglas de negocio: antiguedad_meses
    if "antiguedad_meses" in out.columns:
        out["antiguedad_meses"] = pd.to_numeric(out["antiguedad_meses"], errors="coerce")
        out.loc[out["antiguedad_meses"] < 0, "antiguedad_meses"] = np.nan

    # 3. Reglas de negocio: edad
    if "edad" in out.columns:
        out["edad"] = pd.to_numeric(out["edad"], errors="coerce")
        invalid_edad = (out["edad"] < 18) | (out["edad"] > 100)
        out.loc[invalid_edad, "edad"] = np.nan

    return out


def impute_missing_values(
    df: pd.DataFrame,
    method_ingreso: str = "conditional_risk",
    k_neighbors: int = 3
) -> pd.DataFrame:
    """Aplica estrategias de imputación fundamentadas estadísticamente:
    1. Imputa 'edad' con la mediana (robusta frente a asimetrías).
    2. Imputa 'antiguedad_meses' con la mediana de valores válidos.
    3. Imputa 'ingreso_mensual' usando:
       - 'conditional_risk': Mediana condicionada al grupo 'nivel_riesgo'.
       - 'knn': Vecinos más cercanos (KNNImputer) sobre variables numéricas.

    Args:
        df: DataFrame con inconsistencias convertidas a NaN.
        method_ingreso: Estrategia para ingreso ('conditional_risk' o 'knn').
        k_neighbors: Número de vecinos para KNNImputer (default: 3).

    Returns:
        pd.DataFrame: DataFrame con valores faltantes imputados.
    """
    out = df.copy()

    # Imputación de edad con la mediana
    if "edad" in out.columns and out["edad"].isnull().any():
        mediana_edad = float(out["edad"].median())
        out["edad"] = out["edad"].fillna(mediana_edad)

    # Imputación de antiguedad_meses con la mediana
    if "antiguedad_meses" in out.columns and out["antiguedad_meses"].isnull().any():
        mediana_antig = float(out["antiguedad_meses"].median())
        out["antiguedad_meses"] = out["antiguedad_meses"].fillna(mediana_antig)

    # Imputación de ingreso_mensual
    if "ingreso_mensual" in out.columns and out["ingreso_mensual"].isnull().any():
        if method_ingreso == "conditional_risk" and "nivel_riesgo" in out.columns:
            # Mediana por estrato de riesgo
            medianas_riesgo = out.groupby("nivel_riesgo")["ingreso_mensual"].transform("median")
            out["ingreso_mensual"] = out["ingreso_mensual"].fillna(medianas_riesgo)
            # En caso de quedar remanente (e.g. grupo completo nulo), fallback a mediana global
            if out["ingreso_mensual"].isnull().any():
                out["ingreso_mensual"] = out["ingreso_mensual"].fillna(out["ingreso_mensual"].median())

        elif method_ingreso == "knn":
            # Codificación ordinal temporal de nivel_riesgo si existe
            num_cols = ["edad", "antiguedad_meses", "ingreso_mensual"]
            temp_df = out.copy()
            if "nivel_riesgo" in temp_df.columns:
                risk_map = {"Bajo": 1, "Medio": 2, "Alto": 3}
                temp_df["riesgo_num"] = temp_df["nivel_riesgo"].map(risk_map)
                num_cols.append("riesgo_num")

            imputer = KNNImputer(n_neighbors=k_neighbors)
            imputed_arr = imputer.fit_transform(temp_df[num_cols])
            imputed_df = pd.DataFrame(imputed_arr, columns=num_cols, index=temp_df.index)
            out["ingreso_mensual"] = imputed_df["ingreso_mensual"]
        else:
            raise ValueError(f"Método de imputación no reconocido: {method_ingreso}")

    return out


def calculate_iqr_bounds(
    series: pd.Series,
    multiplier: float = 1.5
) -> Dict[str, float]:
    """Calcula los límites estadísticos del método del Rango Intercuartílico (IQR).

    IQR = Q3 - Q1
    Límite Superior = Q3 + 1.5 * IQR
    Límite Inferior = Q1 - 1.5 * IQR

    Args:
        series: Serie numérica de pandas.
        multiplier: Factor de escala del IQR (default: 1.5 de Tukey).

    Returns:
        Dict[str, float]: Diccionario con Q1, Q3, IQR, limite_inferior y limite_superior.
    """
    valid_series = series.dropna()
    q1 = float(valid_series.quantile(0.25))
    q3 = float(valid_series.quantile(0.75))
    iqr = q3 - q1
    lower_bound = q1 - (multiplier * iqr)
    upper_bound = q3 + (multiplier * iqr)

    return {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "limite_inferior": lower_bound,
        "limite_superior": upper_bound,
    }


def treat_outliers_iqr(
    df: pd.DataFrame,
    column: str = "ingreso_mensual",
    action: str = "winsorize",
    multiplier: float = 1.5
) -> Tuple[pd.DataFrame, List[int], Dict[str, float]]:
    """Detecta y trata outliers en la columna especificada según el método IQR.

    Args:
        df: DataFrame de entrada.
        column: Nombre de la columna numérica a evaluar.
        action: Técnica a aplicar:
            - 'winsorize': Acotamiento de valores extremos al Límite Superior / Inferior.
            - 'eliminate': Eliminación fundamentada de las filas atípicas.
        multiplier: Multiplicador de IQR (default: 1.5).

    Returns:
        Tuple[pd.DataFrame, List[int], Dict[str, float]]:
            - DataFrame tratado.
            - Lista de índices de registros atípicos identificados.
            - Diccionario con los parámetros del IQR.
    """
    bounds = calculate_iqr_bounds(df[column], multiplier=multiplier)
    lower = bounds["limite_inferior"]
    upper = bounds["limite_superior"]

    outlier_mask = (df[column] < lower) | (df[column] > upper)
    outlier_indices = df.index[outlier_mask].tolist()

    out = df.copy()
    if action == "winsorize":
        out[column] = out[column].clip(lower=lower, upper=upper)
    elif action == "eliminate":
        out = out.loc[~outlier_mask].copy().reset_index(drop=True)
    else:
        raise ValueError(f"Acción de tratamiento desconocida: {action}")

    return out, outlier_indices, bounds


# ==============================================================================
# FASE 4: COMPARATIVA FINAL Y PERSISTENCIA
# ==============================================================================

def compute_comparative_metrics(
    df_raw: pd.DataFrame,
    df_clean: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """Calcula un resumen comparativo de media, mediana y desviación estándar
    entre el conjunto crudo y el conjunto limpio.

    Args:
        df_raw: DataFrame original antes de limpieza.
        df_clean: DataFrame resultante después del pipeline.
        columns: Lista de columnas numéricas a comparar.

    Returns:
        pd.DataFrame: Tabla comparativa estructurada.
    """
    if columns is None:
        columns = ["edad", "ingreso_mensual", "antiguedad_meses"]

    records = []
    for col in columns:
        if col in df_raw.columns and col in df_clean.columns:
            records.append({
                "variable": col,
                "raw_media": float(df_raw[col].mean()),
                "clean_media": float(df_clean[col].mean()),
                "raw_mediana": float(df_raw[col].median()),
                "clean_mediana": float(df_clean[col].median()),
                "raw_std": float(df_raw[col].std()),
                "clean_std": float(df_clean[col].std()),
                "reduccion_std_pct": float(
                    ((df_raw[col].std() - df_clean[col].std()) / df_raw[col].std()) * 100.0
                ) if df_raw[col].std() > 0 else 0.0,
            })

    return pd.DataFrame(records).set_index("variable")


def export_curated_data(
    df_clean: pd.DataFrame,
    conn: sqlite3.Connection,
    table_name: str = "clientes_curados"
) -> int:
    """Exporta el conjunto de datos limpio a una nueva tabla en la base de datos relacional.

    Args:
        df_clean: DataFrame limpio y sin inconsistencias.
        conn: Conexión SQLite activa.
        table_name: Nombre de la tabla de destino (default: 'clientes_curados').

    Returns:
        int: Número de registros persistidos.
    """
    df_clean.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.commit()

    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    return int(count)


# ==============================================================================
# EJECUCIÓN ORQUESTADA DEL PIPELINE COMPLETO
# ==============================================================================

def run_full_pipeline(
    method_ingreso: str = "conditional_risk",
    outlier_action: str = "winsorize"
) -> Dict[str, Any]:
    """Ejecuta el pipeline completo de principio a fin según las 4 fases de la guía:
    1. Ingesta desde SQLite en memoria.
    2. EDA y métricas iniciales.
    3. Normalización, imputación y gestión de outliers por IQR.
    4. Comparativa de métricas y persistencia en tabla 'clientes_curados'.

    Returns:
        Dict[str, Any]: Reporte detallado del pipeline.
    """
    conn = create_sqlite_database(in_memory=True)
    df_raw = extract_raw_data(conn)
    eda_stats = calculate_eda_metrics(df_raw)

    # Fase 3: Limpieza
    df_step1 = clean_inconsistencies(df_raw)
    df_step2 = impute_missing_values(df_step1, method_ingreso=method_ingreso)
    df_clean, outlier_idx, iqr_bounds = treat_outliers_iqr(
        df_step2, column="ingreso_mensual", action=outlier_action
    )

    # Fase 4: Métricas y Persistencia
    comparativa = compute_comparative_metrics(df_raw, df_clean)
    rows_persisted = export_curated_data(df_clean, conn, table_name="clientes_curados")

    return {
        "df_raw": df_raw,
        "df_clean": df_clean,
        "eda_stats": eda_stats,
        "outlier_indices": outlier_idx,
        "iqr_bounds": iqr_bounds,
        "comparativa": comparativa,
        "rows_persisted": rows_persisted,
        "null_count_curated": int(df_clean.isnull().sum().sum()),
        "connection": conn,
    }
