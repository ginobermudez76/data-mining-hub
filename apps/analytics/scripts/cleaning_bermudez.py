"""Módulo de Limpieza y Calidad de Datos - Implementación Estudiante (Unidad 2 - Tema 3).
Autor: Gino Maximiliano Bermúdez Santos
Repositorio: data-mining-hub

Cumple estrictamente con la especificación básica de la práctica (Sección 1.2):
- normalize_frame(df) -> pd.DataFrame
- impute_column(df, column, strategy) -> pd.DataFrame
- treat_outliers(df, column, method, action) -> pd.DataFrame
- run_cleaning_pipeline() -> dict
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sqlalchemy.engine import Engine

from src.db_connector import extract_raw_data, get_database_engine

RAW_TABLE = "customer_credit_transactions"
TARGET_CLEAN_TABLE = "customer_credit_clean_bermudez"
NUMERIC_COLS = ["age", "annual_income", "credit_score", "loan_amount"]


# ==============================================================================
# ACTIVIDAD 1.2: SPEC BÁSICA DE FUNCIONES ANALÍTICAS
# ==============================================================================

def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Estandariza inconsistencias según spec 1.2:
    - region: strip + Title Case.
    - age: valores fuera de [18, 100] -> NaN (para forzar paso a imputación).

    Returns:
        pd.DataFrame: DataFrame normalizado (sin mutar el original).
    """
    out = df.copy()

    # Normalización de region
    if "region" in out.columns:
        out["region"] = out["region"].apply(
            lambda val: str(val).strip().title() if pd.notna(val) else val
        )

    # Regla de negocio en age
    if "age" in out.columns:
        out["age"] = pd.to_numeric(out["age"], errors="coerce")
        invalid_mask = (out["age"] < 18) | (out["age"] > 100)
        out.loc[invalid_mask, "age"] = np.nan

    return out


# Aliases para compatibilidad
clean_inconsistencies = normalize_frame
normalize_inconsistencies = normalize_frame


def impute_column(
    df: pd.DataFrame,
    column: str,
    strategy: str = "median",
    n_neighbors: int = 5,
) -> pd.DataFrame:
    """Imputa los valores NaN en `column` según la estrategia elegida:
    - 'mean': media matemática (μ) para variables numéricas continuas.
    - 'median': mediana (robusta ante asimetría y outliers).
    - 'mode': moda categórica (obligatoria para 'region').
    - 'knn': imputación algorítmica avanzada con KNNImputer (n_neighbors=5).

    Returns:
        pd.DataFrame: DataFrame con la columna imputada (sin mutar el original).
    """
    if column not in df.columns:
        raise KeyError(f"La columna '{column}' no existe en el DataFrame.")

    strategy_lower = strategy.strip().lower()
    out = df.copy()

    if not out[column].isna().any():
        return out

    if strategy_lower == "mean":
        if not pd.api.types.is_numeric_dtype(out[column]):
            raise ValueError(f"Estrategia 'mean' solo aplica a columnas numéricas: {column}")
        out[column] = out[column].fillna(float(out[column].mean()))

    elif strategy_lower == "median":
        if not pd.api.types.is_numeric_dtype(out[column]):
            raise ValueError(f"Estrategia 'median' solo aplica a columnas numéricas: {column}")
        out[column] = out[column].fillna(float(out[column].median()))

    elif strategy_lower == "mode":
        mode_series = out[column].mode(dropna=True)
        if not mode_series.empty:
            out[column] = out[column].fillna(mode_series.iloc[0])

    elif strategy_lower == "knn":
        if not pd.api.types.is_numeric_dtype(out[column]):
            raise ValueError(f"Estrategia 'knn' requiere columna numérica: {column}")

        # Seleccionar todas las numéricas disponibles para modelar relaciones multivariables
        numeric_cols = [c for c in NUMERIC_COLS if c in out.columns and pd.api.types.is_numeric_dtype(out[c])]
        if not numeric_cols or column not in numeric_cols:
            numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()

        valid_samples = int(out[column].notna().sum())
        effective_k = max(1, min(n_neighbors, valid_samples))

        imputer = KNNImputer(n_neighbors=effective_k)
        imputed_numeric_array = imputer.fit_transform(out[numeric_cols])
        imputed_numeric_df = pd.DataFrame(
            imputed_numeric_array, columns=numeric_cols, index=out.index
        )
        out[column] = imputed_numeric_df[column]

    else:
        raise ValueError(
            f"Estrategia '{strategy}' no soportada. Use 'mean', 'median', 'mode' o 'knn'."
        )

    return out


def treat_outliers(
    df: pd.DataFrame,
    column: str,
    method: str = "iqr",
    action: str = "cap",
) -> pd.DataFrame:
    """Detecta y trata outliers según la spec 1.2:
    - Detección: 'zscore' (3σ) e 'iqr' (1.5·IQR).
    - Acción: 'cap' (acotamiento a umbrales para preservar el 100% de filas).

    Returns:
        pd.DataFrame: DataFrame con valores acotados (sin mutar el original).
    """
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        return df.copy()

    method_lower = method.strip().lower()
    action_lower = action.strip().lower()
    out = df.copy()
    s = out[column].dropna()

    if s.empty:
        return out

    if method_lower == "iqr":
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
    elif method_lower in ["zscore", "3sigma"]:
        mean_val = float(s.mean())
        std_val = float(s.std(ddof=1)) if len(s) > 1 else 0.0
        lo = mean_val - 3.0 * std_val
        hi = mean_val + 3.0 * std_val
    else:
        raise ValueError(f"Método '{method}' no soportado. Use 'iqr' o 'zscore'.")

    # En finanzas, límites negativos en ingresos o préstamos se acotan en cero
    if lo < 0:
        lo = 0.0

    if action_lower == "cap":
        out[column] = out[column].clip(lower=lo, upper=hi)
    else:
        raise ValueError(f"Acción '{action}' no soportada en spec básica. Use 'cap'.")

    return out


def cap_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "iqr",
    factor: Optional[float] = None,
    min_zero: bool = True,
) -> pd.DataFrame:
    """Wrapper multi-columna para treat_outliers compatible con suites anteriores."""
    target_cols = columns or ["annual_income", "loan_amount"]
    out = df.copy()
    for col in target_cols:
        out = treat_outliers(out, col, method=method, action="cap")
    return out


handle_outliers = cap_outliers


def impute_missing_values(
    df: pd.DataFrame,
    numeric_strategy: str = "median",
    categorical_strategy: str = "mode",
) -> pd.DataFrame:
    """Imputa sistemáticamente todas las columnas con nulos."""
    out = df.copy()
    for col in out.columns:
        if not out[col].isna().any():
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            out = impute_column(out, col, strategy=numeric_strategy)
        else:
            out = impute_column(out, col, strategy=categorical_strategy)
    return out


# ==============================================================================
# AUDITORÍA DE CALIDAD Y MÉTRICAS ESTADÍSTICAS
# ==============================================================================

def calculate_audit_metrics(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    column: str = "annual_income",
) -> Dict[str, Any]:
    """Calcula y compara métricas descriptivas clave antes y después del pipeline."""
    s_before = df_before[column].dropna()
    s_after = df_after[column].dropna()

    var_before = float(s_before.var()) if len(s_before) > 1 else 0.0
    var_after = float(s_after.var()) if len(s_after) > 1 else 0.0
    var_diff = var_after - var_before
    var_pct_change = (
        ((var_after - var_before) / var_before * 100) if var_before != 0 else 0.0
    )

    return {
        "column": column,
        "records_before": len(df_before),
        "records_after": len(df_after),
        "nulls_before": int(df_before[column].isna().sum()),
        "nulls_after": int(df_after[column].isna().sum()),
        "mean_before": float(s_before.mean()),
        "mean_after": float(s_after.mean()),
        "variance_before": var_before,
        "variance_after": var_after,
        "variance_difference": var_diff,
        "variance_pct_change": var_pct_change,
        "std_before": float(s_before.std()) if len(s_before) > 1 else 0.0,
        "std_after": float(s_after.std()) if len(s_after) > 1 else 0.0,
        "min_before": float(s_before.min()) if not s_before.empty else 0.0,
        "min_after": float(s_after.min()) if not s_after.empty else 0.0,
        "max_before": float(s_before.max()) if not s_before.empty else 0.0,
        "max_after": float(s_after.max()) if not s_after.empty else 0.0,
    }


def print_quality_audit_report(
    df_raw: pd.DataFrame,
    df_clean: pd.DataFrame,
    target_column: str = "annual_income",
) -> None:
    """Genera e imprime el reporte por consola para la evaluación de calidad."""
    metrics = calculate_audit_metrics(df_raw, df_clean, column=target_column)

    print("\n" + "=" * 78)
    print("      REPORTE DE AUDITORÍA DE CALIDAD DE DATOS - BERMÚDEZ (CRISP-DM)")
    print("=" * 78)
    print(f"Columna Auditada: {target_column}")
    print(f"Total Registros Iniciales: {metrics['records_before']}")
    print(f"Total Registros Finales:   {metrics['records_after']} (Retención: 100.0%)")
    print(f"Valores Nulos Iniciales:   {metrics['nulls_before']}")
    print(f"Valores Nulos Finales:     {metrics['nulls_after']}")
    print("-" * 78)
    print(f"{'MÉTRICA':<25} | {'ANTES (Crudo)':<22} | {'DESPUÉS (Limpio)':<22}")
    print("-" * 78)
    print(
        f"{'Media (μ)':<25} | {metrics['mean_before']:>20.2f} | {metrics['mean_after']:>20.2f}"
    )
    print(
        f"{'Varianza (σ²)':<25} | {metrics['variance_before']:>20.2f} | {metrics['variance_after']:>20.2f}"
    )
    print(
        f"{'Desviación Estándar (σ)':<25} | {metrics['std_before']:>20.2f} | {metrics['std_after']:>20.2f}"
    )
    print(
        f"{'Mínimo':<25} | {metrics['min_before']:>20.2f} | {metrics['min_after']:>20.2f}"
    )
    print(
        f"{'Máximo':<25} | {metrics['max_before']:>20.2f} | {metrics['max_after']:>20.2f}"
    )
    print("-" * 78)
    print(
        f"Impacto en Varianza: Δ = {metrics['variance_difference']:+.2f} ({metrics['variance_pct_change']:+.2f}%)"
    )
    print("=" * 78)

    print("\n" + "-" * 78)
    print("REFLEXIÓN TEÓRICO-PRÁCTICA (Riesgo Crediticio y Capping vs. Eliminación):")
    print("-" * 78)
    print(
        "En el contexto de modelado de riesgo crediticio, eliminar directamente los registros\n"
        "con valores atípicos (outliers) en variables críticas como 'annual_income' o 'loan_amount'\n"
        "constituiría un error metodológico severo: se perdería información fundamental sobre los\n"
        "segmentos de clientes de mayor exposición económica (perfiles de alto patrimonio o préstamos\n"
        "corporativos), introduciendo un fuerte sesgo de selección y reduciendo el poder estadístico\n"
        "de la muestra. Al implementar la técnica de Acotamiento (Capping/Winsorization), se limitan\n"
        "los valores extremos a un umbral estadístico controlado (ej. Q3 + 1.5*IQR), lo cual estabiliza\n"
        "la varianza y protege la convergencia de los algoritmos de Machine Learning sin sacrificar\n"
        "registros legítimos del historial transaccional."
    )
    print("-" * 78 + "\n")


# ==============================================================================
# PIPELINE ORQUESTADOR
# ==============================================================================

def run_cleaning_pipeline(
    engine: Optional[Engine] = None,
    df: Optional[pd.DataFrame] = None,
    save_to_db: bool = True,
    target_table: str = TARGET_CLEAN_TABLE,
    source_query: str = f"SELECT * FROM {RAW_TABLE} ORDER BY transaction_id;",
    imputation_strategy: str = "median",
    outlier_method: str = "iqr",
    verbose: bool = True,
) -> dict:
    """Orquesta el pipeline completo de la spec 1.2:
    1. Normalizar (normalize_frame).
    2. Imputar (impute_column por columna).
    3. Tratar outliers (treat_outliers con cap IQR en annual_income y loan_amount).
    4. Persistir en PostgreSQL (to_sql en customer_credit_clean_bermudez).

    Returns:
        dict: Resumen con tabla destino, filas de entrada y salida, y el DataFrame limpio.
    """
    db_engine = engine or get_database_engine()

    if df is not None:
        raw = df.copy()
    else:
        raw = extract_raw_data(source_query, engine=db_engine)

    if verbose:
        print(f"[Pipeline Bermúdez] Iniciando pipeline de limpieza spec 1.2...")
        print(f"[Pipeline Bermúdez] Filas iniciales: {len(raw)}, Columnas: {len(raw.columns)}")

    # 1. Normalización
    clean = normalize_frame(raw)

    # 2. Imputación
    for col in NUMERIC_COLS:
        if col in clean.columns and clean[col].isna().any():
            clean = impute_column(clean, col, strategy=imputation_strategy, n_neighbors=5)
    if "region" in clean.columns and clean["region"].isna().any():
        clean = impute_column(clean, "region", strategy="mode")

    # 3. Outliers (capping IQR en annual_income y loan_amount)
    for col in ["annual_income", "loan_amount"]:
        if col in clean.columns:
            clean = treat_outliers(clean, col, method=outlier_method, action="cap")

    # 4. Persistencia en PostgreSQL
    if save_to_db and db_engine is not None:
        clean.to_sql(target_table, con=db_engine, if_exists="replace", index=False)
        if verbose:
            print(f"[Pipeline Bermúdez] Tabla '{target_table}' guardada exitosamente en PostgreSQL.")

    if verbose:
        print_quality_audit_report(raw, clean, target_column="annual_income")

    return {
        "clean_table": target_table,
        "rows_in": int(len(raw)),
        "rows_out": int(len(clean)),
        "dataframe": clean,
    }


if __name__ == "__main__":
    run_cleaning_pipeline(imputation_strategy="median", outlier_method="iqr")
