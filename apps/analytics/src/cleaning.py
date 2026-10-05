"""Módulo de Limpieza y Calidad de Datos (Unidad 2 - Tema 3).

Implementa un pipeline algorítmico robusto para el tratamiento de la "Tríada
Patológica de Datos":
1. Gestión de Inconsistencias (Normalización de formatos y validación de reglas de negocio).
2. Tratamiento de Valores Faltantes (Imputación estadística por media, mediana, moda y KNN).
3. Manejo Algorítmico de Outliers (Detección por IQR/3-Sigma y Acotamiento/Capping).
4. Orquestación del pipeline y persistencia en PostgreSQL (customer_credit_clean).
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sqlalchemy.engine import Engine

from src.db_connector import extract_raw_data, get_database_engine


# ==============================================================================
# ACTIVIDAD 3.1: GESTIÓN DE INCONSISTENCIAS (NORMALIZACIÓN)
# ==============================================================================

def clean_inconsistencies(
    df: pd.DataFrame,
    region_col: str = "region",
    age_col: str = "age",
    min_age: int = 18,
    max_age: int = 100,
) -> pd.DataFrame:
    """Estandariza formatos categóricos y valida reglas de negocio numéricas.

    - Normaliza la columna `region` eliminando espacios en blanco (strip)
      y convirtiendo a formato Title Case (ej: '  costa ' -> 'Costa').
    - Valida reglas de negocio en `age`: valores fuera del rango [min_age, max_age]
      (por ejemplo, age < 18 o age > 100) son reemplazados estructuralmente
      por NaN para forzar su paso a la fase de imputación.

    Args:
        df: DataFrame de entrada con datos crudos.
        region_col: Nombre de la columna de región a estandarizar.
        age_col: Nombre de la columna de edad para validación de reglas.
        min_age: Límite de edad mínima legal según regla de negocio (defecto: 18).
        max_age: Límite de edad máxima según regla de negocio (defecto: 100).

    Returns:
        pd.DataFrame: Copia del DataFrame con datos normalizados e inconsistencias
                      convertidas a NaN.
    """
    df_clean = df.copy()

    # 1. Normalización de formato en columna categórica 'region'
    if region_col in df_clean.columns:
        df_clean[region_col] = df_clean[region_col].apply(
            lambda val: str(val).strip().title() if pd.notna(val) else val
        )

    # 2. Validación de regla de negocio en variable numérica 'age'
    if age_col in df_clean.columns:
        # Convertir a numérico por seguridad si viniera en otro formato
        df_clean[age_col] = pd.to_numeric(df_clean[age_col], errors="coerce")
        invalid_mask = (df_clean[age_col] < min_age) | (df_clean[age_col] > max_age)
        df_clean.loc[invalid_mask, age_col] = np.nan

    return df_clean


# Alias descriptivo para compatibilidad con la guía
normalize_inconsistencies = clean_inconsistencies


# ==============================================================================
# ACTIVIDAD 3.2: TRATAMIENTO DE VALORES FALTANTES (IMPUTACIÓN)
# ==============================================================================

def impute_column(
    df: pd.DataFrame,
    column: str,
    strategy: str = "median",
    n_neighbors: int = 3,
) -> pd.DataFrame:
    """Imputa valores nulos en una columna específica usando una estrategia dada.

    Soporta 4 estrategias analíticas:
    1. 'mean': Reemplazo por la media matemática (μ) para variables continuas.
    2. 'median': Reemplazo por la mediana (robusta ante asimetría/outliers).
    3. 'mode': Reemplazo por la moda (obligatoria para variables categóricas).
    4. 'knn': Imputación algorítmica avanzada multivariable con KNNImputer.

    Args:
        df: DataFrame de entrada.
        column: Nombre de la columna a imputar.
        strategy: Estrategia ('mean', 'median', 'mode', 'knn').
        n_neighbors: Número de vecinos para KNNImputer (aplica solo si strategy='knn').

    Returns:
        pd.DataFrame: Copia del DataFrame con la columna imputada.

    Raises:
        KeyError: Si la columna no existe en el DataFrame.
        ValueError: Si la estrategia especificada no es soportada.
    """
    if column not in df.columns:
        raise KeyError(f"La columna '{column}' no existe en el DataFrame.")

    strategy_lower = strategy.strip().lower()
    df_result = df.copy()

    # Si no hay nulos en la columna, retornar directamente
    if not df_result[column].isna().any():
        return df_result

    if strategy_lower == "mean":
        if not pd.api.types.is_numeric_dtype(df_result[column]):
            raise ValueError(
                f"La estrategia 'mean' solo aplica a columnas numéricas. Columna '{column}' es {df_result[column].dtype}."
            )
        mean_val = float(df_result[column].mean())
        df_result[column] = df_result[column].fillna(mean_val)

    elif strategy_lower == "median":
        if not pd.api.types.is_numeric_dtype(df_result[column]):
            raise ValueError(
                f"La estrategia 'median' solo aplica a columnas numéricas. Columna '{column}' es {df_result[column].dtype}."
            )
        median_val = float(df_result[column].median())
        df_result[column] = df_result[column].fillna(median_val)

    elif strategy_lower == "mode":
        mode_series = df_result[column].mode()
        if not mode_series.empty:
            mode_val = mode_series.iloc[0]
            df_result[column] = df_result[column].fillna(mode_val)

    elif strategy_lower == "knn":
        if not pd.api.types.is_numeric_dtype(df_result[column]):
            raise ValueError(
                f"La estrategia 'knn' requiere columnas numéricas. Columna '{column}' es {df_result[column].dtype}."
            )

        # Seleccionar todas las columnas numéricas para capturar correlaciones multivariables
        numeric_cols = df_result.select_dtypes(include=[np.number]).columns.tolist()
        
        # Ajustar n_neighbors si el número de muestras completas es menor
        valid_samples = int(df_result[column].notna().sum())
        effective_k = max(1, min(n_neighbors, valid_samples))

        imputer = KNNImputer(n_neighbors=effective_k)
        imputed_numeric_array = imputer.fit_transform(df_result[numeric_cols])
        imputed_numeric_df = pd.DataFrame(
            imputed_numeric_array, columns=numeric_cols, index=df_result.index
        )
        # Actualizar la columna solicitada con el resultado multivariable
        df_result[column] = imputed_numeric_df[column]

    else:
        raise ValueError(
            f"Estrategia '{strategy}' no válida. Opciones soportadas: 'mean', 'median', 'mode', 'knn'."
        )

    return df_result


def impute_missing_values(
    df: pd.DataFrame,
    numeric_strategy: str = "median",
    categorical_strategy: str = "mode",
    n_neighbors: int = 3,
) -> pd.DataFrame:
    """Aplica imputación sistemática a todas las columnas del DataFrame que contengan nulos.

    Args:
        df: DataFrame de entrada.
        numeric_strategy: Estrategia para columnas numéricas ('mean', 'median', 'knn').
        categorical_strategy: Estrategia para columnas categóricas ('mode').
        n_neighbors: Vecinos para KNN si se usa 'knn'.

    Returns:
        pd.DataFrame: DataFrame sin valores faltantes.
    """
    df_imputed = df.copy()
    columns_with_nulls = [c for c in df_imputed.columns if df_imputed[c].isna().any()]

    for col in columns_with_nulls:
        if pd.api.types.is_numeric_dtype(df_imputed[col]):
            df_imputed = impute_column(
                df_imputed,
                col,
                strategy=numeric_strategy,
                n_neighbors=n_neighbors,
            )
        else:
            df_imputed = impute_column(
                df_imputed, col, strategy=categorical_strategy
            )

    return df_imputed


# ==============================================================================
# ACTIVIDAD 3.3: MANEJO ALGORÍTMICO DE OUTLIERS (ACOTADO / CAPPING)
# ==============================================================================

def cap_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "iqr",
    factor: Optional[float] = None,
    min_zero: bool = True,
) -> pd.DataFrame:
    """Detecta y trata valores atípicos mediante la técnica de Acotamiento (Capping/Winsorization).

    En lugar de descartar registros (lo que sesga el modelo y destruye muestras valiosas),
    se fijan umbrales estadísticos y se acotan los valores extremos (clipping), preservando
    el 100% de las filas del dataset.

    Métodos estadísticos de detección soportados:
    - 'iqr': Rango Intercuartílico:
        IQR = Q3 - Q1
        Upper = Q3 + (factor * IQR) [factor por defecto: 1.5]
        Lower = Q1 - (factor * IQR)
    - '3sigma' / 'zscore': Regla empírica del límite:
        Upper = μ + (factor * σ) [factor por defecto: 3.0]
        Lower = μ - (factor * σ)

    Args:
        df: DataFrame de entrada.
        columns: Lista de columnas a tratar (por defecto: ['annual_income', 'loan_amount']).
        method: Método de detección ('iqr', '3sigma', 'zscore').
        factor: Factor multiplicador opcional (1.5 para IQR, 3.0 para 3sigma si es None).
        min_zero: Si es True, asegura que el límite inferior no sea negativo para variables financieras.

    Returns:
        pd.DataFrame: Copia del DataFrame con valores extremos acotados.
    """
    target_columns = columns or ["annual_income", "loan_amount"]
    method_lower = method.strip().lower()
    df_capped = df.copy()

    for col in target_columns:
        if col not in df_capped.columns or not pd.api.types.is_numeric_dtype(
            df_capped[col]
        ):
            continue

        series = df_capped[col].dropna()
        if series.empty:
            continue

        if method_lower == "iqr":
            effective_factor = 1.5 if factor is None else factor
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            upper_limit = q3 + (effective_factor * iqr)
            lower_limit = q1 - (effective_factor * iqr)

        elif method_lower in ["3sigma", "zscore", "std"]:
            effective_factor = 3.0 if factor is None else factor
            mean_val = float(series.mean())
            std_val = float(series.std(ddof=1)) if len(series) > 1 else 0.0
            upper_limit = mean_val + (effective_factor * std_val)
            lower_limit = mean_val - (effective_factor * std_val)

        else:
            raise ValueError(
                f"Método '{method}' no soportado. Use 'iqr' o '3sigma'."
            )

        # En variables financieras (ingreso, monto de préstamo), los valores no pueden ser negativos
        if min_zero and lower_limit < 0:
            lower_limit = 0.0

        # Aplicar Capping (Winsorization) preservando el registro íntegro
        df_capped[col] = df_capped[col].clip(lower=lower_limit, upper=upper_limit)

    return df_capped


# Alias descriptivo
handle_outliers = cap_outliers


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
    print("      REPORTE DE AUDITORÍA DE CALIDAD DE DATOS (CRISP-DM FASE 3)")
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
# ACTIVIDAD 4: INTEGRACIÓN (PIPELINE FINAL ORQUESTADOR)
# ==============================================================================

def run_cleaning_pipeline(
    engine: Optional[Engine] = None,
    df: Optional[pd.DataFrame] = None,
    save_to_db: bool = True,
    target_table: str = "customer_credit_clean",
    source_query: str = "SELECT * FROM customer_credit_transactions ORDER BY transaction_id;",
    imputation_strategy: str = "median",
    outlier_method: str = "iqr",
    verbose: bool = True,
) -> pd.DataFrame:
    """Orquesta secuencialmente el pipeline completo de preparación y calidad de datos.

    Secuencia de ejecución:
    1. Extracción de datos crudos (desde PostgreSQL o DataFrame provisto).
    2. Filtro y resolución de inconsistencias:
       - Normalización de formato de la columna 'region' (Strip y Title Case).
       - Detección de reglas de negocio en 'age' (< 18 o > 100 -> NaN).
    3. Tratamiento de valores faltantes (Imputación estadística/KNN y Moda para categóricos).
    4. Detección y acotamiento de Outliers (Capping en 'annual_income' y 'loan_amount').
    5. Persistencia automatizada en PostgreSQL en la tabla `customer_credit_clean`.

    Args:
        engine: Motor SQLAlchemy para interactuar con PostgreSQL.
        df: DataFrame opcional para ejecutar el pipeline en memoria o pruebas.
        save_to_db: Si es True, persiste el resultado en la base de datos PostgreSQL.
        target_table: Nombre de la tabla destino (defecto: 'customer_credit_clean').
        source_query: Consulta SQL para extraer datos si no se proporciona DataFrame.
        imputation_strategy: Estrategia para variables numéricas ('median', 'mean', 'knn').
        outlier_method: Método de acotamiento de outliers ('iqr', '3sigma').
        verbose: Si es True, imprime mensajes de progreso y auditoría en consola.

    Returns:
        pd.DataFrame: Dataset limpio, consistente, imputado y acotado.
    """
    db_engine = engine or get_database_engine()

    # Paso 0: Obtención del dataset
    if df is not None:
        df_raw = df.copy()
    else:
        df_raw = extract_raw_data(source_query, engine=db_engine)

    if verbose:
        print("[Pipeline] Iniciando proceso de limpieza y calidad de datos...")
        print(f"[Pipeline] Filas iniciales: {len(df_raw)}, Columnas: {len(df_raw.columns)}")

    # Paso 1: Filtro de inconsistencias (normalización de 'region' y reglas de negocio en 'age')
    df_step1 = clean_inconsistencies(
        df_raw, region_col="region", age_col="age", min_age=18, max_age=100
    )

    # Paso 2: Imputación de valores faltantes (numéricos según estrategia y categóricos por moda)
    df_step2 = impute_missing_values(
        df_step1,
        numeric_strategy=imputation_strategy,
        categorical_strategy="mode",
        n_neighbors=3,
    )

    # Paso 3: Detección y Acotamiento de Outliers en annual_income y loan_amount
    df_step3 = cap_outliers(
        df_step2,
        columns=["annual_income", "loan_amount"],
        method=outlier_method,
        factor=1.5,
        min_zero=True,
    )

    df_clean = df_step3

    # Paso 4: Persistencia en PostgreSQL
    if save_to_db and db_engine is not None:
        if verbose:
            print(f"[Pipeline] Guardando datos limpios en la tabla '{target_table}'...")
        df_clean.to_sql(
            name=target_table,
            con=db_engine,
            if_exists="replace",
            index=False,
        )
        if verbose:
            print(f"[Pipeline] Tabla '{target_table}' guardada exitosamente en PostgreSQL.")

    # Generar auditoría de calidad por consola
    if verbose:
        print_quality_audit_report(df_raw, df_clean, target_column="annual_income")

    return df_clean


if __name__ == "__main__":
    # Ejecución directa del pipeline
    run_cleaning_pipeline(imputation_strategy="median", outlier_method="iqr")
