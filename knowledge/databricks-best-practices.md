# Mejores Prácticas de Databricks para Migraciones desde DataStage

Este documento cubre las mejores prácticas específicas de Databricks al migrar desde DataStage.

## Delta Lake - El Fundamento

Delta Lake es crucial para reemplazar capacidades de DataStage datasets y tablas intermedias.

### Ventajas sobre archivos tradicionales
- **ACID Transactions**: Garantiza consistencia
- **Time Travel**: Auditoría y rollback fácil
- **Schema Evolution**: Evolución de esquema automática
- **Unified Batch & Streaming**: Simplifica arquitectura
- **Compaction & Optimization**: Mejor performance

### Operaciones Básicas

```python
# Leer Delta Table
df = spark.read.format("delta").load("/mnt/delta/customers")

# Escribir Delta Table (sobrescribir)
df.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save("/mnt/delta/customers")

# Escribir Delta Table (append)
df.write.format("delta") \
    .mode("append") \
    .save("/mnt/delta/customers")

# Merge (UPSERT) - Reemplaza DataStage Update/Insert stages
from delta.tables import DeltaTable

delta_table = DeltaTable.forPath(spark, "/mnt/delta/customers")
delta_table.alias("target").merge(
    df_updates.alias("updates"),
    "target.customer_id = updates.customer_id"
).whenMatchedUpdateAll() \
 .whenNotMatchedInsertAll() \
 .execute()
```

### Time Travel y Auditoría

```python
# DataStage no tiene equivalente nativo - Delta Lake lo hace fácil

# Leer versión histórica
df_yesterday = spark.read.format("delta") \
    .option("versionAsOf", 5) \
    .load("/mnt/delta/customers")

# O por timestamp
df_yesterday = spark.read.format("delta") \
    .option("timestampAsOf", "2024-01-01") \
    .load("/mnt/delta/customers")

# Ver historial
delta_table = DeltaTable.forPath(spark, "/mnt/delta/customers")
display(delta_table.history())

# Restaurar versión anterior
delta_table.restoreToVersion(5)
```

### Optimización Delta

```python
# Compactar archivos pequeños (DataStage no lo necesita, pero crítico en Databricks)
spark.sql("OPTIMIZE delta.`/mnt/delta/customers`")

# Con Z-Ordering (mejor para queries específicas)
spark.sql("OPTIMIZE delta.`/mnt/delta/customers` ZORDER BY (customer_id, region)")

# Vacuum - limpiar archivos antiguos (cuidado con time travel)
spark.sql("VACUUM delta.`/mnt/delta/customers` RETAIN 168 HOURS")  # 7 días
```

---

## Particionamiento - Crítico para Performance

En DataStage, el particionamiento es más simple. En Spark/Databricks es fundamental.

### Tipos de Particionamiento

**1. Particionamiento Físico (Storage)**
```python
# Particionar por columna al escribir (similar a DataStage partitioning)
df.write.format("delta") \
    .partitionBy("year", "month") \
    .save("/mnt/delta/transactions")

# Beneficio: Queries con filtros en estas columnas serán mucho más rápidas
# WHERE year = 2024 AND month = 1  <- Lee solo esa partición física
```

**2. Particionamiento en Memoria (DataFrame)**
```python
# Repartition - full shuffle (costoso)
df_repartitioned = df.repartition(200)  # 200 particiones
df_repartitioned = df.repartition(200, "customer_id")  # Por columna

# Coalesce - reduce particiones sin shuffle (más eficiente)
df_coalesced = df.coalesce(10)

# Cuándo usar qué:
# - Repartition: Antes de operaciones pesadas (joins, aggregations)
# - Coalesce: Antes de escribir, para reducir número de archivos
```

### Guía de Particionamiento

```python
# ❌ MAL: Demasiadas particiones pequeñas
df.repartition(10000).write...  # Miles de archivos pequeños

# ✅ BIEN: Balance entre paralelismo y overhead
num_cores = spark.sparkContext.defaultParallelism
df.repartition(num_cores * 2).write...

# Para writes, consolidar:
df.repartition(num_cores * 2) \
    .write \
    .coalesce(50) \  # 50 archivos finales
    .save(...)
```

### Migrar DataStage Partitioning

| DataStage Partition | Databricks Equivalent |
|---------------------|----------------------|
| Auto | `df` (sin repartition explícito) |
| Hash | `df.repartition("key_column")` |
| Modulus | `df.repartition(N, "key_column")` |
| Range | `df.repartitionByRange("sort_column")` |
| Round Robin | `df.repartition(N)` |
| Same | No hacer nada (mantener particionamiento) |
| Entire | `df.coalesce(1)` |

---

## Broadcast Joins - Optimización Clave

DataStage Lookup stages → Databricks Broadcast Joins

### Cuándo Usar Broadcast
- Tabla pequeña (< 10GB, idealmente < 1GB)
- Join con tabla grande
- Evita shuffle costoso de la tabla grande

```python
from pyspark.sql.functions import broadcast

# ✅ BIEN: Small reference table
df_result = df_large.join(
    broadcast(df_small_reference),
    on="key",
    how="left"
)

# ❌ MAL: Broadcast de tabla grande
df_result = df_large1.join(
    broadcast(df_large2),  # Falla o muy lento
    on="key"
)
```

### Auto Broadcast Configuration
```python
# Configurar threshold de auto-broadcast
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "100MB")

# Disable auto-broadcast si quieres control manual
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
```

---

## Caching y Persistence

DataStage mantiene datos en memoria implícitamente. En Spark es explícito.

### Cuándo Usar Cache

```python
# Usar cache cuando DataFrame se usa múltiples veces
df_important = df.filter(...).select(...)

# Sin cache - se recomputa cada vez
df_important.count()  # Compute
df_important.show()   # Compute again
df_important.write... # Compute again!

# Con cache - se computa una vez
df_important = df.filter(...).select(...).cache()
df_important.count()  # Compute and cache
df_important.show()   # Read from cache
df_important.write... # Read from cache

# IMPORTANTE: Unpersist cuando termines
df_important.unpersist()
```

### Niveles de Storage

```python
from pyspark import StorageLevel

# Memory only (default para .cache())
df.persist(StorageLevel.MEMORY_ONLY)

# Memory and disk (si no cabe en memoria, spill a disco)
df.persist(StorageLevel.MEMORY_AND_DISK)

# Disk only
df.persist(StorageLevel.DISK_ONLY)

# Con serialización (menor memoria, más CPU)
df.persist(StorageLevel.MEMORY_AND_DISK_SER)
```

---

## PySpark Optimizations

### 1. Evitar UDFs Python Cuando Sea Posible

```python
# ❌ LENTO: UDF Python
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType

@udf(returnType=IntegerType())
def calculate_age_udf(birth_year):
    return 2024 - birth_year

df = df.withColumn("age", calculate_age_udf("birth_year"))

# ✅ RÁPIDO: Native Spark functions
df = df.withColumn("age", F.lit(2024) - F.col("birth_year"))
```

### 2. Usar Built-in Functions

```python
from pyspark.sql import functions as F

# String operations
df.withColumn("name_upper", F.upper(F.col("name")))
df.withColumn("name_clean", F.trim(F.regexp_replace("name", "[^a-zA-Z]", "")))

# Date operations
df.withColumn("year", F.year("date_column"))
df.withColumn("days_diff", F.datediff(F.current_date(), "date_column"))

# Null handling
df.withColumn("clean_value", F.coalesce("column1", "column2", F.lit("default")))

# Conditional logic
df.withColumn("category",
    F.when(F.col("amount") > 1000, "HIGH")
     .when(F.col("amount") > 100, "MEDIUM")
     .otherwise("LOW")
)
```

### 3. Preferir DataFrame API sobre RDD

```python
# ❌ EVITAR: RDD operations
rdd_result = df.rdd.map(lambda row: complex_logic(row))

# ✅ PREFERIR: DataFrame API con expresiones SQL
df_result = df.select(
    "*",
    F.expr("complex SQL expression").alias("result")
)
```

---

## Databricks Notebooks Best Practices

### Estructura Recomendada

```python
# COMMAND ----------
# MAGIC %md
# MAGIC # Job Title
# MAGIC Description, migration date, source DS job

# COMMAND ----------
# MAGIC %md
# MAGIC ## 📝 Parameters

# COMMAND ----------
# Widgets para parámetros (reemplaza DataStage job parameters)
dbutils.widgets.text("input_path", "", "Input Path")
dbutils.widgets.text("output_path", "", "Output Path")
dbutils.widgets.dropdown("load_type", "FULL", ["FULL", "INCREMENTAL"], "Load Type")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 📦 Imports and Configuration

# COMMAND ----------
from pyspark.sql import functions as F
from pyspark.sql.types import *
from delta.tables import DeltaTable
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 🔧 Configuration

# COMMAND ----------
# Spark configurations
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 📊 Data Processing
# MAGIC ### Stage 1: Read Input

# COMMAND ----------
# ... processing code ...

# COMMAND ----------
# MAGIC %md
# MAGIC ### Stage 2: Transform

# COMMAND ----------
# ... transformation code ...

# COMMAND ----------
# MAGIC %md
# MAGIC ## ✅ Validation and Logging

# COMMAND ----------
# Final validation and metrics
```

### Markdown para Documentación

```python
# COMMAND ----------
# MAGIC %md
# MAGIC ## DataStage Stage Mapping
# MAGIC 
# MAGIC | DataStage Stage | Type | Databricks Implementation |
# MAGIC |----------------|------|---------------------------|
# MAGIC | Input_Customers | Sequential File | spark.read.csv() |
# MAGIC | Transform_Clean | Transformer | withColumn() operations |
# MAGIC | Aggregate_Sales | Aggregator | groupBy().agg() |
# MAGIC | Output_Results | Dataset | Delta Lake write |
```

### Widgets para Parámetros

```python
# Tipos de widgets
dbutils.widgets.text("param_string", "default", "String Parameter")
dbutils.widgets.dropdown("param_choice", "A", ["A", "B", "C"], "Choice Parameter")
dbutils.widgets.combobox("param_combo", "default", ["opt1", "opt2"], "Combo Parameter")
dbutils.widgets.multiselect("param_multi", "A", ["A", "B", "C"], "Multi Select")

# Obtener valores
value = dbutils.widgets.get("param_string")

# Remover widgets (útil al finalizar)
dbutils.widgets.removeAll()
```

---

## Monitoring y Logging

### Logging Estructurado

```python
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log events
logger.info(f"Starting job: {job_name}")
logger.info(f"Processing date: {run_date}")
logger.info(f"Records read: {df.count()}")
logger.warning(f"High reject rate: {reject_rate}%")
logger.error(f"Validation failed: {error_message}")

# Exception handling con logging
try:
    result = process_data()
    logger.info("Processing completed successfully")
except Exception as e:
    logger.error(f"Error in processing: {str(e)}", exc_info=True)
    raise
```

### Métricas y Validación

```python
# Capturar métricas
metrics = {
    "job_name": "customer_etl",
    "run_timestamp": datetime.now(),
    "input_records": df_input.count(),
    "output_records": df_output.count(),
    "rejected_records": df_reject.count(),
    "processing_time_seconds": processing_time
}

# Escribir métricas a tabla
metrics_df = spark.createDataFrame([metrics])
metrics_df.write.format("delta") \
    .mode("append") \
    .save("/mnt/metrics/job_metrics")

# Validaciones
assert df_output.count() > 0, "No records in output"
assert df_output.count() == expected_count, f"Count mismatch"

# Data quality checks
null_check = df_output.filter(F.col("key_column").isNull()).count()
assert null_check == 0, f"Found {null_check} null keys"
```

---

## Performance Tuning Tips

### 1. Adaptive Query Execution (AQE)

```python
# Habilitar AQE (recomendado para todas las migraciones)
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
```

### 2. Configuración de Shuffle

```python
# Ajustar particiones de shuffle para tu dataset
spark.conf.set("spark.sql.shuffle.partitions", "200")  # Default

# Guideline: 
# - Small data (< 1GB): 50-100
# - Medium data (1-10GB): 200-400
# - Large data (> 10GB): 400-2000
```

### 3. Evitar Data Skew

```python
# Problema: Una partición mucho más grande que otras
# Solución 1: Añadir salt a la key
df_salted = df.withColumn("salt", (F.rand() * 10).cast("int"))
df_salted = df_salted.withColumn("salted_key", F.concat("key", F.lit("_"), "salt"))

# Join con salted key
df_result = df_salted.join(df_other_salted, on="salted_key")

# Solución 2: Broadcast si una tabla es pequeña
df_result = df_large.join(broadcast(df_small), on="key")

# Solución 3: AQE maneja skew automáticamente
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
```

### 4. Minimizar Shuffles

```python
# ❌ MALO: Múltiples shuffles
df.repartition(100) \
    .groupBy("key") \  # Shuffle 1
    .agg(...) \
    .join(df2, "key") \  # Shuffle 2
    .orderBy("value")  # Shuffle 3

# ✅ MEJOR: Planificar particionamiento
df.repartition(100, "key") \  # Particionar por key una vez
    .groupBy("key") \  # No shuffle (ya particionado por key)
    .agg(...) \
    .join(df2.repartition("key"), "key") \  # Shuffle reducido
    .orderBy("value")
```

---

## Security y Governance

### Databricks Secrets

```python
# NO hacer esto - credentials en código
username = "admin"
password = "password123"  # ❌ NUNCA

# Usar Databricks Secrets
username = dbutils.secrets.get(scope="jdbc-scope", key="username")
password = dbutils.secrets.get(scope="jdbc-scope", key="password")

# Usar en JDBC connection
df = spark.read.format("jdbc") \
    .option("url", jdbc_url) \
    .option("user", username) \
    .option("password", password) \
    .load()
```

### Unity Catalog

```python
# Usar tablas catalogadas en lugar de paths
# ❌ Antiguo: Paths directos
df = spark.read.format("delta").load("/mnt/data/customers")

# ✅ Moderno: Unity Catalog
df = spark.read.table("catalog.schema.customers")

# Write to catalog
df.write.mode("overwrite").saveAsTable("catalog.schema.customers")
```

---

## Testing y Validación

### Unit Tests para Transformaciones

```python
# Crear test data
test_data = [
    (1, "John", "2000-01-01", 100),
    (2, "Jane", None, 200),
    (3, "Bob", "1990-15-99", -50),  # Invalid date
]
test_df = spark.createDataFrame(test_data, ["id", "name", "birth_date", "amount"])

# Aplicar transformación
result_df = apply_transformation(test_df)

# Validar resultados
assert result_df.count() == expected_count
assert result_df.filter(F.col("amount") < 0).count() == 0
```

### Comparación con DataStage Output

```python
# Para validar migración correcta
df_datastage_output = spark.read.csv("/mnt/datastage_output/")
df_databricks_output = spark.read.format("delta").load("/mnt/databricks_output/")

# Comparar schemas
print("DataStage schema:", df_datastage_output.schema)
print("Databricks schema:", df_databricks_output.schema)

# Comparar counts
ds_count = df_datastage_output.count()
db_count = df_databricks_output.count()
print(f"Count difference: {db_count - ds_count}")

# Comparar registros específicos
diff = df_databricks_output.subtract(df_datastage_output)
print(f"Records only in Databricks: {diff.count()}")
```

---

## Migration Checklist

Cuando migres un DataStage job, verifica:

✅ **Funcionalidad**
- [ ] Todas las transformaciones migradas
- [ ] Todas las validaciones implementadas
- [ ] Error handling equivalente
- [ ] Logging apropiado

✅ **Performance**
- [ ] Particionamiento optimizado
- [ ] Broadcast joins donde apropiado
- [ ] Caching para DataFrames reutilizados
- [ ] AQE habilitado

✅ **Data Quality**
- [ ] Validaciones de schema
- [ ] Null handling correcto
- [ ] Validaciones de counts
- [ ] Checksums si necesario

✅ **Operacional**
- [ ] Parámetros como widgets
- [ ] Secrets para credentials
- [ ] Logging estructurado
- [ ] Métricas capturadas
- [ ] Documentación clara

✅ **Testing**
- [ ] Tests con data de muestra
- [ ] Comparación con output DataStage
- [ ] Validación end-to-end
- [ ] Performance testing
