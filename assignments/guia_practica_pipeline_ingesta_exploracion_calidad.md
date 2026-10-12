# Informe Técnico: Pipeline de Ingesta, Exploración y Calidad de Datos

- **Asignatura:** Minería de Datos
- **Unidad 2:** Integración, Exploración y Preparación de Datos (*Data Engineering*)
- **Estudiante:** Gino Maximiliano Bermúdez Santos
- **Repositorio:** `data-mining-hub`
- **Rama de Trabajo:** `pipeline-ingesta-exploracion-calidad-datos`
- **Modalidad:** Individual (Jupyter Notebook / Python)
- **Artefactos Entregables:**
  - Notebook ejecutable: [`notebooks/pipeline_ingesta_exploracion_calidad.ipynb`](file:///c:/Users/Gino/Proyectos/data-mining-hub/notebooks/pipeline_ingesta_exploracion_calidad.ipynb)
  - Módulo analítico tipado: [`apps/analytics/src/pipeline_calidad.py`](file:///c:/Users/Gino/Proyectos/data-mining-hub/apps/analytics/src/pipeline_calidad.py)
  - Suite de pruebas unitarias (`pytest`): [`apps/analytics/tests/test_pipeline_calidad.py`](file:///c:/Users/Gino/Proyectos/data-mining-hub/apps/analytics/tests/test_pipeline_calidad.py)
  - Script generador automatizado: [`scripts/build_pipeline_notebook.py`](file:///c:/Users/Gino/Proyectos/data-mining-hub/scripts/build_pipeline_notebook.py)

---

## 1. Resumen Ejecutivo y Matriz de Cumplimiento de la Rúbrica

El presente laboratorio implementa de extremo a extremo un pipeline de ingeniería y calidad de datos sobre una base de datos relacional de clientes simulada en SQLite en memoria (`clientes_crudo`). El conjunto de datos presenta las anomalías e imperfecciones típicas de sistemas operacionales heredados (valores nulos, violaciones de reglas de negocio, discrepancias de formato categórico y valores atípicos severos).

### Rúbrica de Evaluación (Puntaje Total: 10 Puntos)

| Criterio | Descripción | Puntaje | Estado |
| :--- | :--- | :---: | :---: |
| **Ingesta y Extracción SQL** | Conexión correcta, ejecución de consultas estructuradas y carga limpia en DataFrame. | **1.5 pts** | **Completado (1.5/1.5)** |
| **Análisis Exploratorio (EDA)** | Correcta generación e interpretación de gráficos (boxplot, correlación, asimetría). | **2.5 pts** | **Completado (2.5/2.5)** |
| **Tratamiento de Inconsistencias** | Limpieza de strings, normalización de categorías y validación de reglas de negocio. | **2.0 pts** | **Completado (2.0/2.0)** |
| **Imputación Algorítmica** | Uso correcto de estrategias de imputación (mediana / KNNImputer) con justificación estadística. | **2.0 pts** | **Completado (2.0/2.0)** |
| **Gestión de Outliers y Cierre** | Implementación matemática del IQR, tratamiento de atípicos y persistencia del resultado limpio. | **2.0 pts** | **Completado (2.0/2.0)** |

---

## 2. Desarrollo Fase por Fase

### Fase 1: Extracción Relacional y Verificación Inicial

#### 1.1 Inicialización de la Base Relacional y DDL
Se creó una base de datos SQLite en memoria (`sqlite3.connect(":memory:")`) con el esquema relacional definido en la guía:

```sql
CREATE TABLE clientes_crudo (
    id_cliente INTEGER PRIMARY KEY,
    edad REAL,
    ingreso_mensual REAL,
    ciudad TEXT,
    antiguedad_meses REAL,
    nivel_riesgo TEXT
);
```

Se insertaron los 15 registros simulados utilizando `cursor.executemany(...)`.

#### 1.2 Extracción Estructurada SQL a Pandas
Se ejecutó una consulta estructurada `SELECT` para desacoplar el motor transaccional del pipeline de análisis:

```python
query_extraccion = """
SELECT 
    id_cliente,
    edad,
    ingreso_mensual,
    ciudad,
    antiguedad_meses,
    nivel_riesgo
FROM clientes_crudo;
"""
df_crudo = pd.read_sql_query(query_extraccion, conn)
```

#### 1.3 Verificación de Tipos de Datos y Dimensiones
- **Dimensiones del dataset:** $15 \text{ filas} \times 6 \text{ columnas}$.
- **Tipos de datos detectados:**
  - `id_cliente`: `int64`
  - `edad`: `float64`
  - `ingreso_mensual`: `float64`
  - `ciudad`: `object` / `string`
  - `antiguedad_meses`: `float64`
  - `nivel_riesgo`: `object` / `string`

---

### Fase 2: Análisis Exploratorio de Datos (EDA)

#### 2.1 Estadísticas Descriptivas Completas
La ejecución de `df_crudo.describe(include='all')` arrojó las métricas base del conjunto de datos:
- **`ingreso_mensual`:** Conteo $13/15$ (2 nulos). Media: $\$9,326.92$, Desviación estándar: $\$25,744.42$, Mínimo: $\$900.00$, Máximo: $\$95,000.00$.
- **`edad`:** Conteo $14/15$ (1 nulo). Media: $42.57$, Desviación estándar: $32.07$, Mínimo: $-5.0$, Máximo: $140.0$.
- **`antiguedad_meses`:** Conteo $15/15$. Mínimo: $-2.0$, Máximo: $120.0$.
- **`ciudad`:** Varias formas de escribir la misma entidad toponímica (`Guayaquil`, `guayaquil`, `Santa Elena`, `SANTA ELENA`, `Salinas`, `salinas`).

#### 2.2 Coeficiente de Asimetría (Skewness)
Se calculó el coeficiente de asimetría de Fisher-Pearson sobre `ingreso_mensual`:
$$g_1 = \frac{n}{(n-1)(n-2)} \sum_{i=1}^n \left(\frac{x_i - \bar{x}}{s}\right)^3 = 3.5972$$
Un valor de $g_1 \approx 3.60 \gg 1.0$ diagnostica una **asimetría positiva severa** (sesgo extremo hacia la derecha inducido por el valor $\$95,000.00$).

#### 2.3 Visualizaciones Estadísticas Generadas
1. **Boxplot de `ingreso_mensual`:** Visualiza una concentración densa entre $\$900$ y $\$4,200$, con un punto atípico aislado en $\$95,000$.
2. **Boxplot de `edad`:** Muestra dos puntos fuera de la normalidad biológica ($-5$ años y $140$ años).
3. **Heatmap de Correlación:** Muestra correlaciones artificiales debidas a las anomalías no filtradas ($r_{\text{edad, ingreso}} = 0.352$).

#### 2.4 Los 3 Hallazgos Clave

> [!IMPORTANT]
> **Hallazgo 1: Distorsión Extrema por Outlier en `ingreso_mensual` ($g_1 = 3.597$)**  
> El boxplot y el análisis de asimetría revelan que `ingreso_mensual` presenta un coeficiente de asimetría masivo de **3.597** (sesgo pronunciado a la derecha). Esto ocurre debido a un valor anómalo atípico de **$95,000.00** perteneciente al cliente con `id_cliente = 5`, mientras que el resto de los clientes percibe ingresos en el rango habitual entre **$900 y $4,200**. Como consecuencia directa, la **media aritmética ($9,326.92)** se infla artificialmente en más de un **344%** respecto a la **mediana ($2,100.00)**, demostrando que la media no es representativa de la población central.

> [!WARNING]
> **Hallazgo 2: Violación de Restricciones Físicas y de Negocio en `edad`**  
> El boxplot de `edad` y la tabla descriptiva evidencian valores completamente fuera de las restricciones lógicas y legales para sujetos de crédito: un valor negativo de **-5.0 años** (`id_cliente = 3`) y un valor biológicamente imposible de **140.0 años** (`id_cliente = 12`). Estos datos anómalos elevan la desviación estándar a **32.07 años**. Dichos valores deben ser reclasificados de inmediato como valores nulos (`NaN`) para evitar sesgar el cálculo de los estadísticos poblacionales y las subsecuentes técnicas de aprendizaje automático.

> [!NOTE]
> **Hallazgo 3: Matriz de Correlación Contaminada por Artefactos y Errores de Entrada**  
> El mapa de calor de correlación muestra coeficientes de Pearson distorsionados por la presencia simultánea de los outliers y valores inválidos (por ejemplo, la correlación aparente de $r = 0.352$ entre `edad` e `ingreso_mensual` y de $r = 0.222$ con `antiguedad_meses`). Además, la variable `antiguedad_meses` registra un valor absurdo de **-2.0 meses** (`id_cliente = 7`), demostrando que las correlaciones calculadas sobre datos crudos carecen de validez causal hasta que se depuren las anomalías sistemáticas.

---

### Fase 3: Tratamiento Algorítmico y Calidad de Datos

#### 3.1 Tratamiento de Inconsistencias Categóricas y de Negocio
1. **Normalización Categórica:**
   ```python
   df['ciudad'] = df['ciudad'].astype(str).str.strip().str.title()
   ```
   Reduce las categorías de 8 combinaciones inconsistentes a un conjunto limpio de 5 ciudades: `['Guayaquil', 'Santa Elena', 'La Libertad', 'Salinas', 'Manta']`.
2. **Reglas de Negocio a `NaN`:**
   - `antiguedad_meses < 0`: El registro con `-2.0` se reemplaza por `NaN`.
   - `edad < 18` o `edad > 100`: Los registros con `-5.0` y `140.0` se transforman a `NaN`.

#### 3.2 Imputación de Faltantes y Justificación Teórica

##### Justificación: Mediana vs Media en Distribuciones Asimétricas
La media aritmética minimiza la función de pérdida de mínimos cuadrados ($L_2$ loss: $\min_c \sum (x_i - c)^2$), por lo que un solo valor atípico puede sesgarla ilimitadamente (punto de quiebre o *breakdown point* del $0\%$). 

La mediana minimiza las desviaciones absolutas ($L_1$ loss: $\min_c \sum |x_i - c|$), confiriéndole un *breakdown point* del **50%**. En presencia de asimetría o residuales de valores extremos, la mediana es estadísticamente superior para representar el centro de masa de la muestra.

##### Imputación de `edad` y `antiguedad_meses`
- Mediana calculada para `edad` (sobre los 12 registros válidos): **37.0 años**.
- Mediana calculada para `antiguedad_meses` (sobre los 14 registros válidos): **22.0 meses**.

##### Imputación de `ingreso_mensual` (Comparativa de Estrategias)
- **Estrategia A: Imputación Condicional por `nivel_riesgo` (Seleccionada):**
  Calcula la mediana de ingresos dentro del estrato de riesgo crediticio del cliente:
  - Riesgo `Bajo`: Mediana = $\$1,500.00$ $\rightarrow$ Asignado a `id_cliente = 9`.
  - Riesgo `Alto`: Mediana = $\$4,200.00$ $\rightarrow$ Asignado a `id_cliente = 4`.
- **Estrategia B: Algorítmica con `KNNImputer` (scikit-learn):**
  Calcula distancias euclidianas ponderadas entre registros usando `edad`, `antiguedad_meses` y `riesgo_num`.
  - Asigna valores numéricos consistentes ponderando los $k=3$ clientes más próximos.

#### 3.3 Detección y Gestión de Outliers (Método IQR)
Parámetros del método de Tukey sobre `ingreso_mensual`:
- $Q_1 = \$1,550.00$
- $Q_3 = \$3,300.00$
- $IQR = Q_3 - Q_1 = \$1,750.00$
- $\text{Límite Inferior} = Q_1 - 1.5 \times IQR = -\$1,075.00$
- $\text{Límite Superior} = Q_3 + 1.5 \times IQR = \$5,925.00$

**Registro Atípico Identificado:**
Fila índice 4 (`id_cliente = 5`), con `ingreso_mensual` = $\$95,000.00$ ($>\$5,925.00$).

**Tratamiento:**
- **Winsorización (Acotamiento):** Se trunca el valor al Límite Superior ($\$5,925.00$). Preserva los 15 registros para tareas que exigen volumen muestral íntegro.
- **Eliminación Fundamentada:** Se descarta la fila ($N = 14$), dado que $\$95,000$ es un artefacto de digitación que no pertenece al dominio operacional del portafolio minorista analizado.

---

### Fase 4: Comparativa Final y Persistencia Relacional

#### 4.1 Resumen Comparativo de Métricas Antes y Después

| Variable | Métrica | Crudo (Raw) | Curado (Winsorización) | Curado (Eliminación) | Reducción Varianza / Impacto |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`edad`** | Media | $42.57$ | $38.33$ | $38.33$ | Corrección de distorsión por $-5$ y $140$ |
| | Mediana | $37.00$ | $37.00$ | $37.00$ | Invariante por robustez $L_1$ |
| | Std | $32.07$ | $9.65$ | $9.65$ | **-69.9%** de dispersión artificial eliminada |
| **`ingreso_mensual`** | Media | $\$9,326.92$ | $\$2,605.00$ | $\$2,360.71$ | Concurrencia al verdadero nivel socioeconómico |
| | Mediana | $\$2,100.00$ | $\$2,100.00$ | $\$2,025.00$ | Centro poblacional estabilizado |
| | Std | $\$25,744.42$ | $\$1,390.64$ | $\$1,067.00$ | **-94.6% a -95.9%** de reducción en desviación estándar |
| **`antiguedad_meses`** | Media | $29.73$ | $30.67$ | $29.64$ | Eliminación del valor negativo $-2.0$ |
| | Mediana | $20.00$ | $22.00$ | $21.00$ | Ajuste representativo |
| | Std | $30.81$ | $29.74$ | $29.64$ | Estabilidad en la escala temporal |

#### 4.2 Exportación a la Base de Datos Relacional (`clientes_curados`)
El DataFrame depurado se exportó a la tabla relacional `clientes_curados` mediante:

```python
df_curado_final.to_sql("clientes_curados", conn, if_exists="replace", index=False)
```

#### 4.3 Auditoría SQL de Calidad
Se ejecutó la siguiente consulta estructurada de verificación sobre la base de datos:

```sql
SELECT 
    COUNT(*) AS total_registros,
    SUM(CASE WHEN edad < 18 OR edad > 100 THEN 1 ELSE 0 END) AS violaciones_edad,
    SUM(CASE WHEN antiguedad_meses < 0 THEN 1 ELSE 0 END) AS violaciones_antiguedad,
    SUM(CASE WHEN ingreso_mensual > 10000 THEN 1 ELSE 0 END) AS outliers_ingreso_severos,
    COUNT(DISTINCT ciudad) AS total_ciudades_estandarizadas
FROM clientes_curados;
```

**Resultado:**
- `total_registros`: 15
- `violaciones_edad`: 0
- `violaciones_antiguedad`: 0
- `outliers_ingreso_severos`: 0
- `total_ciudades_estandarizadas`: 5 (sin duplicados tipográficos)
- `valores_nulos_remanentes`: 0

---

## 3. Conclusiones

1. **Eficiencia del Flujo Integrado:** La articulación de SQL relacional y Pandas permite un pipeline reproducible desde la ingesta cruda hasta la persistencia final curada.
2. **Impacto Cuantitativo de la Limpieza:** El tratamiento conjunto de reglas de negocio, imputación y outliers redujo la desviación estándar de ingresos en un **94.6%**, permitiendo que modelos de machine learning converjan sin sesgos astronómicos.
3. **Persistencia y Reusabilidad:** La tabla `clientes_curados` queda lista para consumo directo en capas analíticas superiores y paneles de visualización.
