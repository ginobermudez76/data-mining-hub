"""Script generador y ejecutor del Jupyter Notebook de la Práctica de Laboratorio.
Ejecuta todas las celdas y genera el archivo notebooks/practica_preprocesamiento_limpieza.ipynb
con todos los outputs, tablas y reportes visibles.
"""

import io
import json
import os
import sys
import traceback


def run_and_build():
    notebook_cells = []
    cell_exec_counter = 1
    shared_globals = {}

    def add_markdown(source_text):
        lines = [line + "\n" for line in source_text.strip().split("\n")]
        # Remove trailing newline from last line for clean format
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        notebook_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": lines
        })

    def add_code(code_text):
        nonlocal cell_exec_counter
        code_lines = [line + "\n" for line in code_text.strip().split("\n")]
        if code_lines:
            code_lines[-1] = code_lines[-1].rstrip("\n")

        # Capture stdout
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output

        execution_error = None
        result_repr = None

        try:
            # Check if last line is an expression to display
            stmts = code_text.strip().split("\n")
            last_line = stmts[-1].strip()

            if (
                not last_line.startswith("#")
                and not last_line.startswith("import ")
                and not last_line.startswith("from ")
                and not last_line.startswith("print(")
                and not "=" in last_line.split("#")[0]
                and not last_line.startswith("for ")
                and not last_line.startswith("def ")
                and not last_line.startswith("if ")
            ):
                exec_code = "\n".join(stmts[:-1])
                if exec_code.strip():
                    exec(exec_code, shared_globals)
                eval_res = eval(last_line, shared_globals)
                if eval_res is not None:
                    result_repr = repr(eval_res)
            else:
                exec(code_text, shared_globals)

        except Exception as e:
            execution_error = traceback.format_exc()
        finally:
            sys.stdout = old_stdout

        captured_stdout = redirected_output.getvalue()
        outputs = []

        if captured_stdout:
            stdout_lines = [line + "\n" for line in captured_stdout.split("\n")]
            if stdout_lines and stdout_lines[-1] == "\n":
                stdout_lines.pop()
            outputs.append({
                "name": "stdout",
                "output_type": "stream",
                "text": stdout_lines
            })

        if result_repr:
            repr_lines = [line + "\n" for line in result_repr.split("\n")]
            if repr_lines:
                repr_lines[-1] = repr_lines[-1].rstrip("\n")
            outputs.append({
                "data": {
                    "text/plain": repr_lines
                },
                "execution_count": cell_exec_counter,
                "metadata": {},
                "output_type": "execute_result"
            })

        if execution_error:
            err_lines = [line + "\n" for line in execution_error.split("\n")]
            outputs.append({
                "ename": "Exception",
                "evalue": str(e),
                "output_type": "error",
                "traceback": err_lines
            })
            print(f"Error in cell {cell_exec_counter}:\n{execution_error}")

        notebook_cells.append({
            "cell_type": "code",
            "execution_count": cell_exec_counter,
            "metadata": {},
            "outputs": outputs,
            "source": code_lines
        })
        cell_exec_counter += 1

    # =========================================================================
    # CONSTRUCCIÓN DEL CONTENIDO DEL NOTEBOOK
    # =========================================================================

    add_markdown("""# Práctica de Laboratorio: Preprocesamiento y Limpieza de Datos para Machine Learning
**Asignatura:** Minería de Datos / Machine Learning  
**Modalidad:** Aprendizaje en contacto con el Docente  
**Herramienta:** Jupyter Notebook / JupyterLab  
**Estudiante:** Gino Maximiliano Bermúdez Santos  
**Dataset:** `dataset_limpieza_datos.csv` (10,000 registros, 8 columnas)  

---

## 1. Objetivo de la Práctica
Desarrollar habilidades prácticas en la preparación de datos crudos (*raw data*) para su posterior uso en modelos de entrenamiento de Machine Learning. Al finalizar, el estudiante será capaz de identificar anomalías y tratar el mismo conjunto de datos utilizando dos metodologías distintas de la industria:
1. **Enfoque 1 (Pandas):** Tratamiento manual, iterativo y exploratorio paso a paso.
2. **Enfoque 2 (Scikit-Learn Pipelines):** Automatización modular orientada a producción e inferencia en tiempo real mediante `Pipeline` y `ColumnTransformer`.""")

    add_code("""import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Configuración de visualización en Pandas
pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 1000)

print("Librerías importadas correctamente.")""")

    add_markdown("""---
# Fase 1: Exploración Inicial (EDA Básico)

En esta fase se realiza una primera inspección del conjunto de datos crudo para evaluar dimensiones, tipos de datos, presencia de valores faltantes y anomalías numéricas o tipográficas.""")

    add_code("""# Carga del dataset desde la carpeta de datos
possible_paths = [
    "data/dataset_limpieza_datos.csv",
    "../data/dataset_limpieza_datos.csv",
    "apps/analytics/data/dataset_limpieza_datos.csv",
    "dataset_limpieza_datos.csv"
]

dataset_path = None
for p in possible_paths:
    if os.path.exists(p):
        dataset_path = p
        break

if not dataset_path:
    raise FileNotFoundError("No se encontró dataset_limpieza_datos.csv en las rutas configuradas.")

df_raw = pd.read_csv(dataset_path)
print(f"Dataset cargado exitosamente desde: {dataset_path}")
print(f"Dimensiones iniciales: {df_raw.shape[0]} registros x {df_raw.shape[1]} columnas.\\n")
print("Primeros 5 registros:")
print(df_raw.head())""")

    add_code("""print("=== ESTRUCTURA DEL DATAFRAME (.info()) ===")
df_raw.info()""")

    add_code("""print("=== CONTEO Y PORCENTAJE DE VALORES NULOS POR COLUMNA ===")
null_counts = df_raw.isnull().sum()
null_pct = (df_raw.isnull().mean() * 100).round(2)
null_summary = pd.DataFrame({"Nulos": null_counts, "Porcentaje (%)": null_pct})
print(null_summary)""")

    add_code("""print("=== RESUMEN ESTADÍSTICO DE VARIABLES NUMÉRICAS (.describe()) ===")
print(df_raw.describe().round(2))""")

    add_code("""print("=== VALORES ÚNICOS EN VARIABLES CATEGÓRICAS ===")
for col in ["genero", "ciudad", "nivel_educacion"]:
    print(f"\\n--- Conteo de categorías en '{col}': ---")
    print(df_raw[col].value_counts(dropna=False))""")

    add_code("""print("=== DIAGNÓSTICO DE VALORES NEGATIVOS E ILÓGICOS ===")
print(f"Edades negativas (< 0): {(df_raw['edad'] < 0).sum()} registros (Mínimo: {df_raw['edad'].min()})")
print(f"Edades irreales (> 100): {(df_raw['edad'] > 100).sum()} registros (Máximo: {df_raw['edad'].max()})")
print(f"Ingresos negativos (< 0): {(df_raw['ingresos'] < 0).sum()} registros (Mínimo: {df_raw['ingresos'].min()})")
print(f"Hijos negativos (< 0): {(df_raw['hijos'] < 0).sum()} registros (Mínimo: {df_raw['hijos'].min()})")
print(f"Hijos con valores decimales: {((df_raw['hijos'] % 1 != 0) & df_raw['hijos'].notna()).sum()} registros")""")

    add_markdown("""### Reporte de Diagnóstico Inicial sobre la Calidad del Dataset (Fase 1)

Tras la exploración inicial descriptiva, se identificaron las siguientes **5 anomalías principales**:

1. **Presencia Significativa de Valores Faltantes (NaN):**
   - 7 de las 8 columnas del dataset contienen valores nulos.
   - Las variables con mayor afectación son categóricas: `nivel_educacion` (984 nulos, ~9.84%) y `ciudad` (842 nulos, ~8.42%).
   - Las variables numéricas presentan aproximadamente entre un 2.7% y 4.7% de faltantes (`ingresos`: 473, `edad`: 467, `hijos`: 413, `genero`: 408, `altura`: 269).
2. **Valores Ilógicos y Negativos en Variables Cuantitativas:**
   - `edad`: Se detectaron **176 registros negativos** (mínimo: -20 años) y valores biológicamente imposibles superiores a 100 años (máximo: 199 años).
   - `ingresos`: Existen **266 registros con ingresos negativos** (mínimo: -\$49,555.25), lo cual viola la regla de dominio económico.
   - `hijos`: Se encontraron **177 registros con valores negativos** (mínimo: -5 hijos).
3. **Inconsistencia de Tipos y Naturaleza de Datos:**
   - La columna `hijos` tiene **288 registros con valores decimales** (por ejemplo, `1.7979` y `1.8558`), violando la naturaleza discreta del número de descendientes.
   - Al contener valores `NaN`, Pandas castea automáticamente las columnas enteras (`edad`, `hijos`) a `float64`.
4. **Inconsistencias Semánticas, Espacios Extra y Errores Tipográficos:**
   - `genero`: Espacios en blanco accidentales (`'masculino '`, `' femenino'`) y abreviaturas heterogéneas (`'M'`, `'F'`).
   - `ciudad`: Errores de normalización toponímica (`'Sto. Domingo'` vs `'Santo Domingo'`) y espacios residuales (`'Esmeraldas '`).
   - `nivel_educacion`: Múltiples variantes ortográficas para el mismo nivel educativo (`'phd'`, `'Ph.D'`, `'PhD'`, `'master'`, `'Master'`, `'mre'`, `'bachiller'`, `'bachillers'`, `'sin  educacion'`).
5. **Valores Atípicos Extremos (Outliers):**
   - En `ingresos`, el valor máximo alcanza **\$1,991,668.00**, alejándose drásticamente de la mediana (\$59,522.41) y la media (\$70,804.32), lo cual evidencia una fuerte asimetría positiva (*right-skewed*).
   - En `altura`, se observan alturas irreales de hasta **3.97 metros**.""")

    add_markdown("""---
# Fase 2: Enfoque 1 - Tratamiento Manual e Iterativo (Pandas)

En esta sección, se aplican las transformaciones paso a paso modificando directamente una copia del DataFrame (`df_manual`). Este enfoque es ideal para el análisis ad-hoc y la comprensión profunda de cada regla de negocio.""")

    add_code("""# 1. Copia independiente para el Enfoque 1
df_manual = df_raw.copy()

print("Conteo de anomalías antes de la corrección:")
print(f" - Edades < 0: {(df_manual['edad'] < 0).sum()}")
print(f" - Edades > 100: {(df_manual['edad'] > 100).sum()}")
print(f" - Ingresos < 0: {(df_manual['ingresos'] < 0).sum()}")
print(f" - Hijos < 0: {(df_manual['hijos'] < 0).sum()}")

# Reemplazo de valores negativos e ilógicos por NaN para forzar su imputación
df_manual.loc[df_manual['edad'] < 0, 'edad'] = np.nan
df_manual.loc[df_manual['edad'] > 100, 'edad'] = np.nan
df_manual.loc[df_manual['ingresos'] < 0, 'ingresos'] = np.nan
df_manual.loc[df_manual['hijos'] < 0, 'hijos'] = np.nan

# Corrección de hijos decimales: redondear al entero más cercano
df_manual.loc[df_manual['hijos'].notna(), 'hijos'] = df_manual.loc[df_manual['hijos'].notna(), 'hijos'].round()

print("\\nVerificación tras reemplazo por NaN:")
print(f" - Edades < 0: {(df_manual['edad'] < 0).sum()}")
print(f" - Edades > 100: {(df_manual['edad'] > 100).sum()}")
print(f" - Ingresos < 0: {(df_manual['ingresos'] < 0).sum()}")
print(f" - Hijos < 0: {(df_manual['hijos'] < 0).sum()}")""")

    add_code("""# 2. Mapeo y Normalización de Strings en variables categóricas

# A. Normalización de 'genero'
df_manual['genero'] = df_manual['genero'].astype(str).str.strip().str.lower()
mapa_genero = {
    'masculino': 'Masculino',
    'femenino': 'Femenino',
    'm': 'Masculino',
    'f': 'Femenino',
    'nan': np.nan,
    'none': np.nan,
    '': np.nan
}
df_manual['genero'] = df_manual['genero'].map(mapa_genero)

# B. Normalización de 'ciudad'
df_manual['ciudad'] = df_manual['ciudad'].astype(str).str.strip().str.title()
mapa_ciudad = {
    'Sto. Domingo': 'Santo Domingo',
    'Nan': np.nan,
    'None': np.nan,
    '': np.nan
}
df_manual['ciudad'] = df_manual['ciudad'].replace(mapa_ciudad)

# C. Normalización de 'nivel_educacion'
# Limpieza de espacios múltiples internos
df_manual['nivel_educacion'] = (
    df_manual['nivel_educacion']
    .astype(str)
    .str.strip()
    .str.lower()
    .replace(r'\\s+', ' ', regex=True)
)
mapa_educacion = {
    'phd': 'Doctorado',
    'ph.d': 'Doctorado',
    'master': 'Maestría',
    'mre': 'Maestría',
    'bachiller': 'Bachillerato',
    'bachillers': 'Bachillerato',
    'secundaria': 'Secundaria',
    'sin educacion': 'Sin Educación',
    'nan': np.nan,
    'none': np.nan,
    '': np.nan
}
df_manual['nivel_educacion'] = df_manual['nivel_educacion'].map(mapa_educacion)

print("Categorías homologadas y unificadas:")
print("\\nGénero:", df_manual['genero'].value_counts(dropna=False).to_dict())
print("\\nCiudad:", df_manual['ciudad'].value_counts(dropna=False).to_dict())
print("\\nNivel Educación:", df_manual['nivel_educacion'].value_counts(dropna=False).to_dict())""")

    add_code("""# 3. Imputación Manual de Valores Nulos

# Variables numéricas: Imputación con la MEDIANA
columnas_numericas = ['edad', 'ingresos', 'altura', 'hijos']
for col in columnas_numericas:
    mediana_val = df_manual[col].median()
    df_manual[col] = df_manual[col].fillna(mediana_val)
    print(f"Imputación numérica -> '{col}': mediana = {round(mediana_val, 2)}")

# Variables categóricas: Imputación con la MODA
columnas_categoricas = ['genero', 'ciudad', 'nivel_educacion']
for col in columnas_categoricas:
    moda_val = df_manual[col].mode().iloc[0]
    df_manual[col] = df_manual[col].fillna(moda_val)
    print(f"Imputación categórica -> '{col}': moda = '{moda_val}'")

# Casteo estricto a números enteros para 'edad' e 'hijos'
df_manual['edad'] = df_manual['edad'].astype(int)
df_manual['hijos'] = df_manual['hijos'].astype(int)

print("\\nVerificación final de nulos en df_manual:", df_manual.isnull().sum().sum())
print("Tipos de datos resultantes:")
print(df_manual[['edad', 'hijos', 'ingresos', 'altura']].dtypes)""")

    add_markdown("""### Justificación Teórica: ¿Por qué usar la Mediana en lugar de la Media?

En presencia de distribuciones asimétricas o con **valores atípicos extremos (outliers)** (como los ingresos de hasta \$1,991,668.00 detectados en el dataset), la media aritmética se desplaza sensiblemente hacia la cola superior, sobrestimando el valor central típico. La **mediana** es una medida de tendencia central robusta (*resistente*), ya que representa el percentil 50 y no se ve afectada por la magnitud de observaciones extremas.""")

    add_code("""# 4. Estandarización Manual: Fórmula Z = (x - μ) / σ

def estandarizar_serie(serie: pd.Series) -> pd.Series:
    \"\"\"Calcula la estandarización Z-score de forma manual y vectorizada.\"\"\"
    mu = serie.mean()
    sigma = serie.std()
    return (serie - mu) / sigma

# Aplicar a la columna de ingresos
df_manual['ingresos_estandarizados'] = estandarizar_serie(df_manual['ingresos'])

# Verificación de propiedades estadísticas de Z (media ≈ 0, desviación estándar ≈ 1)
z_mean = df_manual['ingresos_estandarizados'].mean()
z_std = df_manual['ingresos_estandarizados'].std()

print("=== VERIFICACIÓN DE ESTANDARIZACIÓN MANUAL (Fase 2) ===")
print(f"Media de Z (esperada ≈ 0.0): {z_mean:.6f}")
print(f"Desviación estándar de Z (esperada ≈ 1.0): {z_std:.6f}")
print("\\nPrimeros 5 registros con ingresos y su versión estandarizada:")
print(df_manual[['ingresos', 'ingresos_estandarizados']].head())""")

    add_markdown("""---
# Fase 3: Enfoque 2 - Automatización con Pipelines (Scikit-Learn)

En esta sección se implementa un flujo automatizado y reproducible utilizando `scikit-learn`. Se vuelve a cargar el dataset original crudo para simular un escenario productivo donde nuevos lotes de datos deben transformarse de forma modular y sin fuga de datos (*data leakage*).""")

    add_code("""# Carga independiente del dataset crudo para la Fase 3
df_pipeline = pd.read_csv(dataset_path)

# Tratamiento inicial de anomalías lógicas de negocio
df_pipeline.loc[df_pipeline['edad'] < 0, 'edad'] = np.nan
df_pipeline.loc[df_pipeline['edad'] > 100, 'edad'] = np.nan
df_pipeline.loc[df_pipeline['ingresos'] < 0, 'ingresos'] = np.nan
df_pipeline.loc[df_pipeline['hijos'] < 0, 'hijos'] = np.nan

# Normalización básica de strings
for col in ['genero', 'ciudad', 'nivel_educacion']:
    df_pipeline[col] = df_pipeline[col].astype(str).str.strip().replace({'nan': np.nan, 'None': np.nan, '': np.nan})

# Definición de listas separadas para variables numéricas y categóricas
numeric_features = ['edad', 'ingresos', 'altura', 'hijos']
categorical_features = ['genero', 'ciudad', 'nivel_educacion']

print("Variables numéricas seleccionadas:", numeric_features)
print("Variables categóricas seleccionadas:", categorical_features)""")

    add_code("""# Construcción de Pipelines y ColumnTransformer

# 1. Pipeline Numérico: SimpleImputer (mediana) + StandardScaler
num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 2. Pipeline Categórico: SimpleImputer (moda) + OneHotEncoder (dummies con puntos extra)
cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 3. ColumnTransformer integrado
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, numeric_features),
        ('cat', cat_pipeline, categorical_features)
    ],
    remainder='drop'  # Ignora columnas como 'n'
)

print("Pipeline ColumnTransformer configurado exitosamente:")
print(preprocessor)""")

    add_code("""# Ejecución del pipeline con .fit_transform()
processed_data = preprocessor.fit_transform(df_pipeline)

# Extracción de los nombres de columnas resultantes
feature_names = preprocessor.get_feature_names_out()

# Creación del DataFrame procesado final
clean_column_names = [col.replace('num__', '').replace('cat__', '') for col in feature_names]
df_processed_final = pd.DataFrame(processed_data, columns=clean_column_names)

print(f"Pipeline ejecutado exitosamente.")
print(f"Dimensiones del dataset final: {df_processed_final.shape[0]} filas x {df_processed_final.shape[1]} columnas.")
print(f"Valores nulos residuales: {df_processed_final.isnull().sum().sum()}\\n")
print("Primeros 5 registros del DataFrame procesado con Scikit-Learn:")
print(df_processed_final.head())""")

    add_markdown("""---
# Pregunta de Reflexión Final y Justificación de Ingeniería

### ¿Cuál de los dos enfoques presentados considera más adecuado si el día de mañana la empresa recibe 50,000 registros nuevos cada hora para predecir en tiempo real? Justifique su respuesta.

**Respuesta: El Enfoque 2 (Automatización con Pipelines y ColumnTransformer de Scikit-Learn)** es el único viable y técnicamente adecuado para un entorno productivo de alta concurrencia e inferencia en tiempo real. 

Las razones arquitectónicas y metodológicas que fundamentan esta decisión son:

1. **Prevención Estricta de Fuga de Datos (*Data Leakage*):**
   - En el Enfoque 1 (Pandas manual), calcular `.median()` o `mean()` sobre el conjunto completo o recalcularlo en cada nuevo lote altera las referencias estadísticas con datos que el modelo no conocía durante el entrenamiento.
   - En el Enfoque 2, el objeto `Pipeline` aprende los parámetros estadísticos ($\mu$, $\sigma$, medianas y categorías del encoder) **únicamente en el conjunto de entrenamiento** (`.fit()`) y luego los aplica de forma congelada e inmutable (`.transform()`) sobre cada nuevo lote de 50,000 registros, garantizando validez estadística y rigor metodológico.

2. **Inferencia en Tiempo Real y Baja Latencia:**
   - La manipulación imperativa de Pandas implica operaciones lentas de interpretación en Python (`.apply()`, condicionales fila por fila, diccionarios manuales).
   - El `Pipeline` de Scikit-Learn está optimizado a nivel C/Cython y vectorizado con NumPy, procesando decenas de miles de filas en milisegundos. Además, puede serializarse en disco (`joblib.dump(preprocessor, 'pipeline.joblib')`) y cargarse instantáneamente en un microservicio de inferencia (por ejemplo, con FastAPI o Triton Inference Server).

3. **Consistencia Dimensional y Manejo de Categorías Nuevas:**
   - Si en un nuevo lote de 50,000 registros aparece una categoría imprevista (por ejemplo, una nueva ciudad como `"Cuenca"` o `"Loja"`), el `OneHotEncoder(handle_unknown='ignore')` la ignora asignando ceros sin detener la ejecución. En contraste, los mapeos manuales de Pandas generarían `NaN` accidentales o desalinearían las columnas necesarias por el algoritmo de Machine Learning (*shape mismatch error*).

4. **Mantenibilidad, MLOps y Reproducibilidad:**
   - Un `ColumnTransformer` encapsula todo el preprocesamiento en un solo objeto reutilizable, facilitando el versionado del modelo, la integración continua (CI/CD) y el monitoreo de deriva de datos (*Data Drift*).""")

    # Estructura JSON final del Notebook
    notebook_dict = {
        "cells": notebook_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    out_path = "notebooks/practica_preprocesamiento_limpieza.ipynb"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=1, ensure_ascii=False)

    print(f"\\n¡Jupyter Notebook generado y ejecutado exitosamente en: {out_path}!")
    print(f"Total de celdas creadas: {len(notebook_cells)}")


if __name__ == "__main__":
    run_and_build()
