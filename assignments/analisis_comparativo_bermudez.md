# Asignación U2-T3: Análisis Comparativo de Limpieza de Datos e Ingeniería sobre Código Existente

**Asignatura:** Minería de Datos (Séptimo Semestre)  
**Institución:** Universidad Estatal Península de Santa Elena (UPSE) - FACSISTEL  
**Estudiante:** Gino Maximiliano Bermúdez Santos  
**Repositorio:** `data-mining-hub`  
**Fecha:** Octubre 2026  

---

## 1. Parte A — Reimplementación Comparativa y Análisis de Tablas

Conforme a la especificación básica de la práctica (Sección 1.2), se implementó el módulo independiente:
```bash
apps/analytics/scripts/cleaning_bermudez.py
```
Este pipeline ejecuta secuencialmente las firmas simples de la especificación:
1. `normalize_frame(df) -> pd.DataFrame`
2. `impute_column(df, column, strategy) -> pd.DataFrame`
3. `treat_outliers(df, column, method, action) -> pd.DataFrame`
4. `run_cleaning_pipeline() -> dict`

El resultado fue materializado en la tabla dedicada de staging:
```sql
customer_credit_clean_bermudez
```

---

### 1.1 Comparativa de Filas: `customer_credit_clean_bermudez` vs `customer_credit_clean`

| Métrica / Dimensión | Spec Básica (`customer_credit_clean_bermudez`) | Pipeline de Producción (`customer_credit_clean`) |
|---|---|---|
| **Filas de Entrada** | 9 (o $N$ filas de la tabla cruda) | 9 (o $N$ filas de la tabla cruda) |
| **Filas de Salida** | 9 filas (100% retención estricta) | 9 filas en seed base / $N - \text{dups}$ en datos masivos |
| **Deduplicación** | No implementada (conserva duplicados) | Activa: elimina duplicados exactos excluyendo PK y timestamps |
| **Validación de Rangos** | Únicamente en `age` $\in [18, 100]$ | Multivariable: `age`, `annual_income`, `credit_score`, `loan_amount` |
| **Persistencia** | `to_sql('customer_credit_clean_bermudez')` | `to_sql('customer_credit_clean')` |
| **Contrato de Retorno** | `pd.DataFrame` en cada función | `tuple[pd.DataFrame, dict]` (DataFrame + reporte de auditoría) |

#### ¿Coinciden las filas? ¿Por qué difieren?
- En el dataset base de 9 registros, la cantidad de filas coincide (9 filas) porque no existían duplicados exactos en el seed inicial.
- Sin embargo, en un entorno de producción o con datos sintéticos sembrados (por ejemplo, con 2017 filas crudas):
  - `customer_credit_clean` arroja **2009 filas** (elimina 8 filas duplicadas exactas).
  - `customer_credit_clean_bermudez` arrojaría **2017 filas**, dado que la especificación básica no contempla el paso de deduplicación.

---

### 1.2 ¿Qué hace la versión del repositorio que la versión básica no hace?

1. **Observabilidad por Paso (Reportes Estructurados):**  
   Cada función analítica en `src/cleaning.py` devuelve una tupla `(pd.DataFrame, dict)` que cuantifica exactamente qué se modificó (cuántos nulos se imputaron, estadísticas descriptivas `before` y `after`, y conteo de outliers detectados). La versión básica opera como una "caja negra" que únicamente transforma el DataFrame sin reportar métricas intermedias.
2. **Deduplicación Automatizada:**  
   La función `_dedup_columns()` identifica y purga filas redundantes antes del procesamiento estadístico, evitando que transacciones duplicadas sesguen la media, mediana y varianza.
3. **Validación Multivariable de Rangos de Negocio (`VALID_RANGES`):**  
   Mientras que la spec básica solo valida `age` entre 18 y 100 años, la versión del repo valida adicionalmente `annual_income` (0 a 300,000), `credit_score` (300 a 850) y `loan_amount` (0 a 100,000), convirtiendo cualquier valor absurdo a `NaN` para su posterior imputación.
4. **Múltiples Acciones ante Outliers:**  
   Soporta `cap` (acotamiento), `drop` (eliminación) y `log` (transformación logarítmica suavizada con `np.log1p`), permitiendo evaluar diferentes tratamientos según el modelo predictivo.
5. **Auditoría de Calidad Global (`audit_frame`):**  
   Genera diagnósticos completos por columna para ser consumidos directamente por la API REST y la interfaz gráfica web.

---

### 1.3 ¿Qué pasaría en producción si nuestra versión básica hubiera reemplazado a la del repositorio?

Si un desarrollador hubiera sobrescrito `apps/analytics/src/cleaning.py` con las firmas simples `-> pd.DataFrame`:

1. **Rotura Inmediata del Backend (FastAPI `src/api.py`):**  
   Los endpoints `/api/cleaning/preview/imputation`, `/api/cleaning/preview/outliers` y `/api/cleaning/preview/normalization` esperan recibir tuplas `clean, report = func(...)`. Si la función solo retorna un DataFrame, la aplicación fallaría con un error crítico en tiempo de ejecución: `TypeError: cannot unpack non-iterable DataFrame object`.
2. **Inutilización del Frontend Web (React + Vite):**  
   La interfaz de usuario en `apps/web/src/features/limpieza/index.jsx` depende de los objetos JSON de reporte para dibujar las tablas de estadísticas antes/después, los badges de conteo y las alertas de inconsistencias. La interfaz quedaría vacía o mostraría errores de red `500 Internal Server Error`.
3. **Pérdida de Observabilidad y Monitoreo de Datos (Data Drift):**  
   En producción, no se podría detectar si un nuevo lote de datos contiene un incremento repentino de valores nulos o si un proveedor envió registros fuera de rango.

---

## 2. Parte B — Extensión del Módulo: Perfil Ingeniería (Opción B1)

Siguiendo las buenas prácticas de ingeniería de software (*extender contratos sin romper consumidores existentes*), se implementaron tres mejoras clave en `apps/analytics/src/cleaning.py` y se expusieron en la API:

### 2.1 Nuevo Método de Detección: MAD (*Median Absolute Deviation*)
- **Fundamento Matemático:**  
  El Z-Score convencional ($\frac{x - \mu}{\sigma}$) sufre severamente del **efecto de enmascaramiento (*masking*)**: un outlier extremo (por ejemplo, un ingreso de \$500,000 en un conjunto de ingresos de \$30,000 a \$60,000) infla tanto la media aritmética como la desviación estándar $\sigma$, reduciendo artificialmente el valor $|z|$ de modo que el outlier no supera el umbral de 3 desviaciones estándar.
- **Implementación Robusta:**  
  $$\text{MAD} = \text{mediana}(|x - \text{mediana}(x)|)$$
  $$\sigma_{\text{MAD}} = 1.4826 \times \text{MAD} \quad \text{(factor de consistencia para distribución normal)}$$
  $$\text{Outlier Mask} = |x - \text{mediana}(x)| > 3.0 \times \sigma_{\text{MAD}}$$
- **Resultado:** En las pruebas con `_dirty_frame()`, Z-Score detecta **0 outliers** en ingresos de 500k debido al masking, mientras que `mad` detecta con éxito el valor atípico de \$500,000.

### 2.2 Regla de Negocio: Whitelist de Regiones
Se definió la whitelist oficial de regiones permitidas por el negocio:
```python
VALID_REGIONS = {"Costa", "Sierra", "Oriente", "Insular"}
```
En `normalize_frame()`, cualquier región no vacía que tras normalizarse no pertenezca a este conjunto (por ejemplo, `"Madrid"`, `"Galapagos"`) se convierte estructuralmente a `np.nan` y se contabiliza en el reporte bajo `invalid_whitelist`, forzando su imputación por moda.

### 2.3 Parametrización de `run_cleaning_pipeline()`
Se parametrizó la función orquestadora preservando retrocompatibilidad absoluta mediante valores por defecto:
```python
def run_cleaning_pipeline(
    strategy: str = "median",
    outlier_method: str = "iqr",
    outlier_action: str = "cap",
) -> dict:
```
Esto permite al usuario o analista alternar entre `strategy="knn"`, `outlier_method="mad"`, o `outlier_action="log"` sin romper ninguna llamada heredada en `api.py`.

---

## 3. Evidencia de Ejecución y Pruebas Automatizadas

Al ejecutar la suite de pruebas completa en el contenedor analítico:
```bash
docker exec dm_analytics2 pytest -v
```

**Resultado:**
- **36 pruebas ejecutadas y 36 aprobadas (100% Passing)**:
  - 7 pruebas del API REST (`test_api.py`).
  - 7 pruebas del módulo de limpieza existente y extensiones B1 (`test_cleaning.py`).
  - 4 pruebas de la reimplementación de Bermúdez (`test_cleaning_bermudez.py`).
  - 3 pruebas de conectividad a PostgreSQL (`test_db_connector.py`).
  - 4 pruebas del módulo EDA (`test_eda.py`).
  - 3 pruebas del pipeline Iris base (`test_pipeline.py`).
  - 4 pruebas del generador sintético (`test_seed_synthetic.py`).
  - 4 pruebas de los retos de aula de la Unidad 2 (`test_taller_retos.py`).

Ambos pipelines conviven en el warehouse de PostgreSQL:
1. `customer_credit_clean` (alimenta el frontend y API con métricas enriquecidas).
2. `customer_credit_clean_bermudez` (evidencia el cumplimiento de la especificación básica de la práctica).
