# Criterios de Evaluación Mejorados

## 📊 Pesos Generales

| Métrica | Peso Anterior | Peso Nuevo | Justificación |
|---------|--------------|------------|---------------|
| **Keyword Coverage** | 25% | 15% | Reducido - presencia de palabras no garantiza calidad |
| **Includes Code** | 30% | 15% | Reducido - más importante que el código sea correcto |
| **Well Structured** | 20% | 10% | Reducido - la estructura es menos crítica que la funcionalidad |
| **Category-Specific** | 25% | **60%** | **Aumentado** - evaluación profunda según el tipo de tarea |

## 🎯 Evaluación de Migraciones Completas (full_migration)

Para determinar qué modelo es mejor en migraciones DSX → Databricks, aplicamos **4 niveles de evaluación**:

---

### 📌 NIVEL 1: CORRECCIÓN TÉCNICA (40%)

**Objetivo:** Verificar que el código generado es técnicamente correcto y funcional.

#### 1.1 Sintaxis Válida (10%)
- **Método:** Validación con `ast.parse()` de Python
- **Evaluación:** `bloques_válidos / total_bloques`
- **Criterio de éxito:** Todos los bloques de código deben parsear sin errores
- **Ejemplo de fallo:**
  ```python
  # Sintaxis inválida - punto y coma en Python
  df = spark.read.csv(path);
  ```

#### 1.2 Schema Fidelity (15%)
- **Método:** Extracción de columnas del DSX vs columnas en PySpark
- **Evaluación:** `columnas_coincidentes / total_columnas_dsx`
- **Criterio de éxito:** ≥80% de columnas del DSX deben aparecer en el código
- **Ejemplo DSX:**
  ```xml
  <Column Name="CustomerID" SqlType="INTEGER"/>
  <Column Name="FirstName" SqlType="VARCHAR"/>
  <Column Name="Email" SqlType="VARCHAR"/>
  ```
- **Esperado en PySpark:**
  ```python
  df.select("CustomerID", "FirstName", "Email")
  ```

#### 1.3 Transformaciones Correctas (15%)
- **Método:** Detectar categorías de transformaciones DSX→PySpark
- **Categorías evaluadas:**
  - String operations: `trim`, `upper`, `lower`, `concat`
  - Date operations: `datediff`, `date_add`, `current_date`
  - Null handling: `isNull`, `isNotNull`, `coalesce`, `when`
  - Aggregations: `groupBy`, `agg`, `sum`, `count`
- **Evaluación:** `categorías_encontradas / 4`
- **Ejemplo DSX:**
  ```datastage
  Trim(Upcase(FirstName)) : " " : Trim(Upcase(LastName))
  ```
- **Esperado en PySpark:**
  ```python
  F.concat(F.trim(F.upper(F.col("FirstName"))), F.lit(" "), F.trim(F.upper(F.col("LastName"))))
  ```

---

### 📌 NIVEL 2: COMPLETITUD (30%)

**Objetivo:** Verificar que todos los componentes del DSX fueron migrados.

#### 2.1 Todos los Stages Migrados (15%)
- **Método:** Detectar presencia de operaciones READ + TRANSFORM + WRITE
- **Evaluación:** `stages_presentes / 3`
- **Criterio de éxito:** Los 3 stages deben estar
- **Mapeo DSX → PySpark:**
  - `Sequential_File` (Source) → `spark.read`
  - `Transformer` → `.withColumn`, `.select`, `.filter`
  - `Sequential_File` (Target) → `.write`

#### 2.2 Parámetros Extraídos (10%)
- **Método:** Comparar parámetros del DSX vs `dbutils.widgets`
- **Evaluación:** `parámetros_migrados / total_parámetros_dsx`
- **Ejemplo DSX:**
  ```xml
  <Parameter>
    <Property Name="Name">INPUT_FILE_PATH</Property>
    <Property Name="DefaultValue">/data/input/customers.csv</Property>
  </Parameter>
  ```
- **Esperado en PySpark:**
  ```python
  dbutils.widgets.text("INPUT_FILE_PATH", "/data/input/customers.csv")
  INPUT_FILE_PATH = dbutils.widgets.get("INPUT_FILE_PATH")
  ```

#### 2.3 Validaciones Incluidas (5%)
- **Método:** Detectar constraints del DSX convertidos a validaciones PySpark
- **Evaluación:** Binario (0% o 5%)
- **Indicadores:** `filter`, `isNotNull`, `isNull`, `constraint`
- **Ejemplo DSX:**
  ```xml
  <Constraint>
    <Property Name="Expression">NOT(IsNull(CustomerID))</Property>
  </Constraint>
  ```
- **Esperado en PySpark:**
  ```python
  df_validated = df.filter(F.col("CustomerID").isNotNull())
  ```

---

### 📌 NIVEL 3: BEST PRACTICES (20%)

**Objetivo:** Verificar que el código sigue mejores prácticas de Databricks.

#### 3.1 Delta Lake Usage (8%)
- **Método:** Detectar uso de formato Delta
- **Evaluación:** 
  - 5% por uso básico de Delta (`format("delta")`)
  - 8% por uso avanzado (+ `OPTIMIZE` o `MERGE`)
- **Ejemplo básico:**
  ```python
  df.write.format("delta").mode("overwrite").save(path)
  ```
- **Ejemplo avanzado:**
  ```python
  df.write.format("delta").mode("overwrite").save(path)
  spark.sql(f"OPTIMIZE delta.`{path}` ZORDER BY (customer_id)")
  ```

#### 3.2 Partitioning Strategy (6%)
- **Método:** Detectar estrategia de particionamiento
- **Evaluación:** 
  - 6% por `partitionBy`
  - 3% por `repartition` o `coalesce`
- **Ejemplo:**
  ```python
  df.write.partitionBy("year", "month").format("delta").save(path)
  ```

#### 3.3 Error Handling (6%)
- **Método:** Detectar manejo de errores
- **Indicadores:** `try:`, `except`, `raise`, `assert`, `logger`
- **Evaluación:**
  - 6% si ≥2 indicadores
  - 3% si 1 indicador
- **Ejemplo:**
  ```python
  try:
      df = spark.read.csv(path)
  except Exception as e:
      logger.error(f"Error reading file: {e}")
      raise
  ```

---

### 📌 NIVEL 4: CALIDAD (10%)

**Objetivo:** Evaluar documentación y estructura del notebook.

#### 4.1 Documentación (5%)
- **Indicadores:** `# MAGIC %md`, `##`, `"""`, `'''`
- **Evaluación:**
  - 5% si ≥3 indicadores
  - 3% si 2 indicadores
  - 2% si 1 indicador
- **Ejemplo:**
  ```python
  # MAGIC %md
  # MAGIC ## 1. Read Customer Data
  # MAGIC 
  # MAGIC Este notebook procesa datos de clientes...
  
  """
  Lee archivo CSV de clientes y aplica transformaciones de limpieza
  """
  df = spark.read.csv(path)
  ```

#### 4.2 Estructura Notebook (5%)
- **Método:** Detectar formato de notebook Databricks
- **Evaluación:**
  - 5% por `# Databricks notebook source`
  - 3% por `# MAGIC %md` o `# COMMAND`
- **Ejemplo:**
  ```python
  # Databricks notebook source
  # MAGIC %md
  # MAGIC # Customer ETL Migration
  # MAGIC **DSX Source:** Simple_Customer_ETL.dsx
  
  # COMMAND ----------
  
  # Import libraries
  from pyspark.sql import functions as F
  ```

---

## 🏆 Cómo Determinar el Mejor Modelo

### Criterios de Selección

1. **Score Overall ≥ 90%** en migraciones completas
2. **Sintaxis válida = 100%** (todos los bloques parsean)
3. **Schema fidelity ≥ 80%** (preserva columnas del DSX)
4. **Completitud = 100%** (todos los stages migrados)
5. **Best practices ≥ 70%** (usa Delta Lake + partitioning)

### Matriz de Comparación

| Modelo | Overall | Sintaxis | Schema | Completitud | Best Practices | Calidad |
|--------|---------|----------|--------|-------------|----------------|---------|
| gpt-5.2-codex | ? | ? | ? | ? | ? | ? |
| gpt-5.1-codex | ? | ? | ? | ? | ? | ? |

> **Nota:** Ejecuta `python evaluate_agent.py --mode azure-entra --models gpt-5.2-codex gpt-5.1-codex` para llenar esta tabla.

---

## 📈 Ejemplo de Reporte Detallado

```
  🔍  DESGLOSE DETALLADO: MIGRACIONES COMPLETAS (full_migration)
================================================================================

  📊  Modelo: gpt-5.2-codex
  ----------------------------------------------------------------------------

    Métrica                              Promedio        Peso
    ------------------------------------------------------------
    NIVEL 1: CORRECCIÓN TÉCNICA                           40%
      └─ Sintaxis válida                   100.0%      (10%)
      └─ Schema fidelity                    87.5%      (15%)
      └─ Transformaciones correctas         95.0%      (15%)

    NIVEL 2: COMPLETITUD                                  30%
      └─ Todos los stages                  100.0%      (15%)
      └─ Parámetros extraídos               85.0%      (10%)
      └─ Validaciones                      100.0%       (5%)

    NIVEL 3: BEST PRACTICES                               20%
      └─ Delta Lake                         80.0%       (8%)
      └─ Partitioning                       50.0%       (6%)
      └─ Error handling                     33.3%       (6%)

    NIVEL 4: CALIDAD                                      10%
      └─ Documentación                      80.0%       (5%)
      └─ Estructura notebook               100.0%       (5%)

    ------------------------------------------------------------
    OVERALL (full_migration)                92.3%        100%
```

---

## 🚀 Próximos Pasos

1. **Ejecutar evaluación completa:**
   ```powershell
   python evaluate_agent.py --mode azure-entra --models gpt-5.2-codex gpt-5.1-codex
   ```

2. **Comparar con/sin knowledge base:**
   ```powershell
   # Con knowledge base (default)
   python evaluate_agent.py --mode azure-entra --models gpt-5.1-codex
   
   # Sin knowledge base
   python evaluate_agent.py --mode azure-entra --models gpt-5.1-codex --no-knowledge
   ```

3. **Analizar casos específicos:**
   - Revisar `evaluation_report.json` → `details` → casos con score < 80%
   - Identificar patrones de error (sintaxis, schema, transformaciones)
   - Iterar sobre knowledge base o prompts

4. **Validar notebooks generados:**
   - Copiar código de `response` del JSON
   - Ejecutar en Databricks workspace
   - Verificar que produce output correcto
