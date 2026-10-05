"""Módulo con las soluciones a los retos prácticos del Taller en Clase (Unidad 2 - Tema 1)."""
from typing import Optional
import pandas as pd
from sqlalchemy.engine import Engine
from src.db_connector import extract_raw_data, extract_raw_data_in_chunks


def count_customers_by_region(engine: Optional[Engine] = None) -> pd.DataFrame:
    """Reto 1: Conteo por región usando agregación SQL (GROUP BY region)."""
    query = """
    SELECT region, COUNT(*) AS total_clientes
    FROM customer_credit_transactions
    GROUP BY region
    ORDER BY total_clientes DESC;
    """
    return extract_raw_data(query, engine=engine)


def get_high_risk_customers(
    threshold: int = 650, engine: Optional[Engine] = None
) -> pd.DataFrame:
    """Reto 2: Perfil de riesgo extrayendo clientes con credit_score < threshold."""
    query = f"""
    SELECT customer_id, credit_score, annual_income, has_defaulted
    FROM customer_credit_transactions
    WHERE credit_score < {threshold}
    ORDER BY credit_score ASC;
    """
    return extract_raw_data(query, engine=engine)


def get_costa_customers_by_income(
    engine: Optional[Engine] = None,
) -> pd.DataFrame:
    """Reto 3: Clientes de Costa ordenados descendentemente por ingreso anual."""
    query = """
    SELECT customer_id, region, annual_income, credit_score
    FROM customer_credit_transactions
    WHERE region = 'Costa'
    ORDER BY annual_income DESC NULLS LAST;
    """
    return extract_raw_data(query, engine=engine)


def reconstruct_dataset_from_chunks(
    chunk_size: int = 3, engine: Optional[Engine] = None
) -> pd.DataFrame:
    """Reto 4: Reconstruye y consolida el dataset a partir de lotes usando pd.concat."""
    query = "SELECT * FROM customer_credit_transactions ORDER BY transaction_id;"
    chunks = list(
        extract_raw_data_in_chunks(query, chunk_size=chunk_size, engine=engine)
    )
    return pd.concat(chunks, ignore_index=True)


if __name__ == "__main__":
    print("=== RETO 1: Conteo por Región ===")
    print(count_customers_by_region())

    print("\n=== RETO 2: Perfil de Riesgo (credit_score < 650) ===")
    print(get_high_risk_customers())

    print(
        "\n=== RETO 3: Clientes de Costa ordenados por ingreso descendente ==="
    )
    print(get_costa_customers_by_income())

    print("\n=== RETO 4: Reconstrucción desde Chunks ===")
    df_reconstructed = reconstruct_dataset_from_chunks()
    print(f"Total registros consolidados: {len(df_reconstructed)}")
    print(df_reconstructed[["transaction_id", "customer_id", "region"]])
