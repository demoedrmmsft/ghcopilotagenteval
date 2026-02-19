# Catálogo de Componentes IBM DataStage

Este documento describe todos los stages principales de IBM DataStage y sus características, para facilitar la migración a Databricks.

## Stages de Entrada/Salida (Input/Output)

### Sequential File Stage
**Propósito**: Leer o escribir archivos secuenciales (CSV, texto delimitado, ancho fijo)

**Propiedades Clave**:
- File path
- Format (delimited, fixed-width)
- Column delimiter
- Quote character
- Header rows
- Null field values
- Reject mode

**Metadata**:
```xml
<Property Name="FileFormat">Delimited</Property>
<Property Name="ColumnDelimiter">,</Property>
<Property Name="QuoteCharacter">"</Property>
<Property Name="FirstLineIsColumnNames">True</Property>
```

**Migración PySpark**:
```python
# Lectura
df = spark.read.format("csv") \
    .option("header", "true") \
    .option("delimiter", ",") \
    .option("quote", "\"") \
    .option("nullValue", "") \
    .load(file_path)

# Escritura
df.write.format("csv") \
    .option("header", "true") \
    .option("delimiter", ",") \
    .mode("overwrite") \
    .save(output_path)
```

### Dataset Stage
**Propósito**: Leer/escribir datasets DataStage (formato binario interno)

**Características**:
- Formato optimizado de DataStage
- Preserva metadata y particionamiento
- Usado para datos intermedios

**Migración PySpark**:
```python
# Usar Delta Lake como equivalente moderno
df = spark.read.format("delta").load(path)
df.write.format("delta").mode("overwrite").save(path)
```

### ODBC Stage / Database Stage
**Propósito**: Conectar a bases de datos relacionales

**Propiedades Clave**:
- Connection string
- Table name
- SQL query
- Write mode (Insert, Update, Upsert, Delete)
- Array size (batch size)

**Migración PySpark**:
```python
# Lectura
df = spark.read.format("jdbc") \
    .option("url", jdbc_url) \
    .option("dbtable", table_name) \
    .option("user", username) \
    .option("password", password) \
    .option("driver", driver_class) \
    .load()

# Escritura con upsert (merge)
from delta.tables import DeltaTable

if DeltaTable.isDeltaTable(spark, target_path):
    delta_table = DeltaTable.forPath(spark, target_path)
    delta_table.alias("target").merge(
        df.alias("source"),
        "target.id = source.id"
    ).whenMatchedUpdateAll() \
     .whenNotMatchedInsertAll() \
     .execute()
else:
    df.write.format("delta").save(target_path)
```

---

## Stages de Procesamiento (Processing)

### Transformer Stage
**Propósito**: Aplicar transformaciones, derivaciones y filtros a los datos

**Componentes**:
1. **Derivations** - Crear/modificar columnas
2. **Constraints** - Filtrar filas
3. **Stage Variables** - Variables temporales para cálculos intermedios
4. **Loop Variables** - Procesar arrays/listas

**Expresiones Comunes**:
```basic
' DataStage BASIC expression examples
If IsNull(Column1) Then "DEFAULT" Else Column1
Trim(Column1) : "-" : Trim(Column2)
Column1[1,10]  ' Substring start=1, length=10
Upcase(Column1)
DateFromDaysSince(Days)
```

**Migración PySpark**:
```python
from pyspark.sql import functions as F

df_transformed = df \
    .withColumn("new_column", 
        F.when(F.col("Column1").isNull(), F.lit("DEFAULT"))
         .otherwise(F.col("Column1"))
    ) \
    .withColumn("concatenated",
        F.concat(F.trim(F.col("Column1")), F.lit("-"), F.trim(F.col("Column2")))
    ) \
    .withColumn("substring_col",
        F.substring(F.col("Column1"), 1, 10)  # PySpark usa índice 1-based
    ) \
    .withColumn("uppercase",
        F.upper(F.col("Column1"))
    ) \
    .filter(F.col("amount") > 100)  # Constraint
```

**Stage Variables**:
```python
# DataStage: Stage variable para cálculo intermedio
# StageVar = Column1 + Column2
# DerivedColumn = StageVar * 10

# PySpark: Usar columnas temporales
df = df \
    .withColumn("_stage_var", F.col("Column1") + F.col("Column2")) \
    .withColumn("DerivedColumn", F.col("_stage_var") * 10) \
    .drop("_stage_var")  # Limpiar temporal
```

### Aggregator Stage
**Propósito**: Agrupar y agregar datos (SUM, COUNT, AVG, MIN, MAX, etc.)

**Propiedades**:
- Grouping keys
- Aggregation calculations
- Group ordering

**Ejemplo DataStage**:
```
Group by: CustomerID
Calculations:
  - TotalAmount = SUM(Amount)
  - OrderCount = COUNT()
  - AvgAmount = AVG(Amount)
  - FirstOrderDate = MIN(OrderDate)
  - LastOrderDate = MAX(OrderDate)
```

**Migración PySpark**:
```python
df_aggregated = df.groupBy("CustomerID").agg(
    F.sum("Amount").alias("TotalAmount"),
    F.count("*").alias("OrderCount"),
    F.avg("Amount").alias("AvgAmount"),
    F.min("OrderDate").alias("FirstOrderDate"),
    F.max("OrderDate").alias("LastOrderDate")
)
```

### Join Stage
**Propósito**: Unir múltiples flujos de datos

**Tipos de Join**:
- Inner Join
- Left Outer Join
- Right Outer Join
- Full Outer Join

**Propiedades**:
- Join keys
- Join type
- Multiple inputs (hasta 128)
- Reject links para non-matches

**Migración PySpark**:
```python
# Inner Join
df_result = df_left.join(df_right, on="join_key", how="inner")

# Left Outer con múltiples condiciones
df_result = df_left.join(
    df_right,
    (df_left["key1"] == df_right["key1"]) & 
    (df_left["key2"] == df_right["key2"]),
    how="left"
)

# Join múltiple
df_result = df1 \
    .join(df2, on="key", how="inner") \
    .join(df3, on="key", how="left")
```

### Lookup Stage
**Propósito**: Buscar valores en tabla de referencia (optimizado para tablas pequeñas)

**Características**:
- Tabla principal (input)
- Tabla de referencia (lookup)
- Lookup keys
- Fallo en lookup → default values
- Caching en memoria

**Migración PySpark**:
```python
# Para tablas de referencia pequeñas (<10GB), usar broadcast join
from pyspark.sql.functions import broadcast, coalesce, lit

# Lookup simple
df_result = df_main.join(
    broadcast(df_reference),
    on="lookup_key",
    how="left"
)

# Con valores default para lookup failures
df_result = df_main.alias("main").join(
    broadcast(df_reference).alias("ref"),
    on="lookup_key",
    how="left"
).select(
    "main.*",
    coalesce(F.col("ref.ref_value"), lit("NOT_FOUND")).alias("ref_value")
)
```

### Sort Stage
**Propósito**: Ordenar datos

**Propiedades**:
- Sort keys
- Sort order (ascending/descending)
- Sort mode (stable vs unstable)
- Create key changes column

**Migración PySpark**:
```python
# Sort simple
df_sorted = df.orderBy("column1", "column2")

# Sort con direcciones mixtas
df_sorted = df.orderBy(
    F.col("column1").asc(),
    F.col("column2").desc()
)

# Sort con manejo de nulls
df_sorted = df.orderBy(
    F.col("column1").asc_nulls_last()
)
```

### Funnel Stage
**Propósito**: Combinar múltiples inputs con misma estructura (UNION)

**Tipos**:
- Continuous funnel: Combina streams en tiempo real
- Sort funnel: Combina y ordena
- Sequence funnel: Preserva orden de entrada

**Migración PySpark**:
```python
# Union simple
df_combined = df1.union(df2).union(df3)

# Union con estructura diferente (union by name)
df_combined = df1.unionByName(df2, allowMissingColumns=True)

# Union + ordenamiento
df_combined = df1.union(df2).orderBy("sort_column")
```

### Remove Duplicates Stage
**Propósito**: Eliminar registros duplicados

**Propiedades**:
- Key columns (para determinar duplicados)
- Keep first/last occurrence

**Migración PySpark**:
```python
# Eliminar duplicados por columnas específicas
df_unique = df.dropDuplicates(subset=["key1", "key2"])

# Mantener primer/último registro (requiere ordenamiento)
from pyspark.sql.window import Window

# Mantener último registro
window_spec = Window.partitionBy("key1", "key2").orderBy(F.desc("timestamp"))
df_unique = df.withColumn("row_num", F.row_number().over(window_spec)) \
    .filter(F.col("row_num") == 1) \
    .drop("row_num")
```

### Filter Stage
**Propósito**: Dividir flujo de datos en múltiples salidas basado en condiciones

**Características**:
- Múltiples outputs con condiciones
- Reject link para registros que no coinciden

**Migración PySpark**:
```python
# Dividir en múltiples DataFrames
df_category_a = df.filter(F.col("category") == "A")
df_category_b = df.filter(F.col("category") == "B")
df_others = df.filter(~F.col("category").isin(["A", "B"]))

# O usar when/otherwise para crear flag column
df_categorized = df.withColumn(
    "category_flag",
    F.when(F.col("amount") > 1000, "HIGH")
     .when(F.col("amount") > 100, "MEDIUM")
     .otherwise("LOW")
)
```

### Row Generator Stage
**Propósito**: Generar filas sintéticas

**Propiedades**:
- Number of rows
- Column values (expressions)

**Migración PySpark**:
```python
# Generar rango de números
df_generated = spark.range(start=1, end=1001, step=1).toDF("id")

# Generar con valores calculados
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

df_generated = spark.range(100).select(
    F.col("id").cast("integer"),
    F.lit("Generated").alias("source"),
    F.current_timestamp().alias("created_at")
)
```

### Pivot Stage
**Propósito**: Transponer filas a columnas

**Propiedades**:
- Pivot column
- Grouping columns
- Aggregation

**Migración PySpark**:
```python
# Pivot
df_pivoted = df.groupBy("customer_id", "year").pivot("product_category") \
    .agg(F.sum("amount"))

# Unpivot (reverse pivot)
df_unpivoted = df.selectExpr(
    "customer_id",
    "year",
    "stack(3, 'Category_A', Category_A, 'Category_B', Category_B, 'Category_C', Category_C) as (category, amount)"
)
```

---

## Stages de Calidad de Datos (Data Quality)

### Match Stage
**Propósito**: Identificar registros duplicados o similares (fuzzy matching)

**Migración**: Usar librerías especializadas
```python
# Opción 1: Similaridad básica con Spark
from pyspark.sql.functions import levenshtein, soundex

df_with_similarity = df1.crossJoin(df2) \
    .withColumn("name_distance", levenshtein("df1.name", "df2.name")) \
    .filter(F.col("name_distance") < 3)

# Opción 2: Usar biblioteca especializada (zingg, splink)
# O integrar con Azure Cognitive Services
```

### Standardize Stage
**Propósito**: Estandarizar y limpiar datos (nombres, direcciones, teléfonos)

**Migración**:
```python
# Estandarización manual con UDFs o expresiones
df_standardized = df \
    .withColumn("phone_clean", F.regexp_replace("phone", "[^0-9]", "")) \
    .withColumn("name_upper", F.upper(F.trim(F.col("name")))) \
    .withColumn("email_lower", F.lower(F.trim(F.col("email"))))

# Para casos complejos, considerar Azure Data Quality services
```

---

## Stages de Control (Control Flow)

### Wait Stage
**Propósito**: Esperar hasta que archivo/condición esté disponible

**Migración**:
```python
# En Databricks, usar Jobs orchestration o Delta Live Tables
# O implementar lógica de espera
import time
from pathlib import Path

def wait_for_file(path, timeout=3600, check_interval=60):
    elapsed = 0
    while elapsed < timeout:
        if dbutils.fs.ls(path):  # En Databricks
            return True
        time.sleep(check_interval)
        elapsed += check_interval
    raise TimeoutError(f"File not available after {timeout} seconds")
```

### Surrogate Key Generator Stage
**Propósito**: Generar claves subrogadas únicas

**Migración**:
```python
from pyspark.sql.functions import monotonically_increasing_id, row_number
from pyspark.sql.window import Window

# Opción 1: Monotonically increasing (no garantiza secuencial)
df_with_key = df.withColumn("surrogate_key", monotonically_increasing_id())

# Opción 2: Row number (secuencial dentro de partición)
window_spec = Window.orderBy(F.lit(1))
df_with_key = df.withColumn("surrogate_key", row_number().over(window_spec))

# Opción 3: UUID
from pyspark.sql.functions import expr
df_with_key = df.withColumn("surrogate_key", expr("uuid()"))
```

---

## Stages Especiales

### Change Capture Stage
**Propósito**: Detectar cambios entre datasets (CDC)

**Migración**:
```python
# Usar Delta Lake Change Data Feed
spark.conf.set("spark.databricks.delta.properties.defaults.enableChangeDataFeed", "true")

# Habilitar en tabla
df.write.format("delta") \
    .option("delta.enableChangeDataFeed", "true") \
    .save(path)

# Leer cambios
changes = spark.read.format("delta") \
    .option("readChangeFeed", "true") \
    .option("startingVersion", 0) \
    .load(path)
```

### Checksum Stage
**Propósito**: Calcular checksums para validación de datos

**Migración**:
```python
from pyspark.sql.functions import md5, sha2, concat_ws

# Checksum de fila
df_with_checksum = df.withColumn(
    "row_checksum",
    md5(concat_ws("|", *df.columns))
)

# Checksum de dataset
from pyspark.sql.functions import sum as _sum, hash

dataset_checksum = df.select(_sum(hash(*df.columns))).collect()[0][0]
```

---

## Consideraciones de Migración

### Performance
- **DataStage partitioning** → **Spark partitioning**: Reparticionar apropiadamente
- **Sort stages**: Minimizar sorts, usar solo cuando necesario
- **Aggregations**: Spark maneja mejor con particionado correcto

### Manejo de Errores
- **Reject links** → Usar filtros y separar DataFrames
- **Warning/Error handling** → Implementar con try/catch y logging

### Metadata
- **Job parameters** → Databricks widgets o configuración
- **Environment variables** → Databricks Secrets o variables de cluster

### Testing
- Validar counts en cada stage
- Comparar checksums con DataStage output
- Verificar transformaciones críticas con casos de prueba
