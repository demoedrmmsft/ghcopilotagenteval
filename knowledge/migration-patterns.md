# Patrones de Migración DataStage → Databricks

Este documento contiene patrones comunes de migración con ejemplos detallados.

## Patrón 1: ETL Simple (Extract-Transform-Load)

### DataStage Job Structure
```
Sequential_File_Input → Transformer → Sequential_File_Output
```

### Componentes DataStage
1. **Sequential File** (Input): Lee CSV con clientes
2. **Transformer**: 
   - Limpia nombres (Trim, Uppercase)
   - Calcula edad desde fecha nacimiento
   - Filtra clientes activos
3. **Sequential File** (Output): Escribe resultado

### Código PySpark Migrado

```python
# Databricks notebook source
# MAGIC %md
# MAGIC # Customer ETL - Migrado desde DataStage
# MAGIC Procesa archivo de clientes, limpia datos y filtra activos

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import *

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parámetros

# COMMAND ----------

dbutils.widgets.text("input_path", "/mnt/raw/customers.csv", "Input Path")
dbutils.widgets.text("output_path", "/mnt/processed/customers", "Output Path")

input_path = dbutils.widgets.get("input_path")
output_path = dbutils.widgets.get("output_path")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 1: Read Input (Sequential_File_Input)

# COMMAND ----------

# DataStage Stage: Sequential_File_Input
# Propiedades: CSV delimitado, header, quote="

df_raw = spark.read.format("csv") \
    .option("header", "true") \
    .option("delimiter", ",") \
    .option("quote", "\"") \
    .option("inferSchema", "true") \
    .load(input_path)

print(f"Registros leídos: {df_raw.count()}")
display(df_raw.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 2: Transform (Transformer)

# COMMAND ----------

# DataStage Stage: Transformer
# Derivations:
#   - CleanName = Trim(Upcase(FirstName)) : " " : Trim(Upcase(LastName))
#   - Age = YearsSince(BirthDate)
# Constraints:
#   - Status = "ACTIVE"

df_transformed = df_raw \
    .withColumn("CleanName", 
        F.concat(
            F.trim(F.upper(F.col("FirstName"))),
            F.lit(" "),
            F.trim(F.upper(F.col("LastName")))
        )
    ) \
    .withColumn("Age",
        F.floor(F.months_between(F.current_date(), F.col("BirthDate")) / 12)
    ) \
    .filter(F.col("Status") == "ACTIVE")

print(f"Registros después de transformación: {df_transformed.count()}")
display(df_transformed.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 3: Write Output (Sequential_File_Output)

# COMMAND ----------

# DataStage Stage: Sequential_File_Output
# Formato: CSV con header

df_transformed.write.format("csv") \
    .option("header", "true") \
    .mode("overwrite") \
    .save(output_path)

print(f"✅ Datos escritos en: {output_path}")
```

---

## Patrón 2: Join con Lookup

### DataStage Job Structure
```
Sequential_File (Orders) ────┐
                              ├→ Join → Lookup (Products) → Output
Sequential_File (Customers) ──┘
```

### Migración PySpark

```python
# COMMAND ----------
# MAGIC %md
# MAGIC ## Leer Fuentes de Datos

# COMMAND ----------

# DataStage: Sequential_File - Orders
df_orders = spark.read.format("csv") \
    .option("header", "true") \
    .load("/mnt/raw/orders.csv")

# DataStage: Sequential_File - Customers
df_customers = spark.read.format("csv") \
    .option("header", "true") \
    .load("/mnt/raw/customers.csv")

# DataStage: Sequential_File - Products (tabla de referencia)
df_products = spark.read.format("csv") \
    .option("header", "true") \
    .load("/mnt/raw/products.csv")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Stage 1: Join Orders con Customers

# COMMAND ----------

# DataStage Stage: Join_1
# Type: Inner Join
# Keys: Orders.CustomerID = Customers.CustomerID

df_joined = df_orders.alias("o").join(
    df_customers.alias("c"),
    on=F.col("o.CustomerID") == F.col("c.CustomerID"),
    how="inner"
).select(
    "o.*",
    F.col("c.CustomerName"),
    F.col("c.CustomerSegment")
)

print(f"Registros después de join: {df_joined.count()}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Stage 2: Lookup Products (Broadcast Join)

# COMMAND ----------

# DataStage Stage: Lookup_Products
# Type: Lookup (tabla pequeña de referencia)
# Keys: ProductID
# Nota: Usando broadcast join para optimizar

from pyspark.sql.functions import broadcast, coalesce

df_enriched = df_joined.alias("main").join(
    broadcast(df_products).alias("prod"),
    on=F.col("main.ProductID") == F.col("prod.ProductID"),
    how="left"
).select(
    "main.*",
    coalesce(F.col("prod.ProductName"), F.lit("UNKNOWN")).alias("ProductName"),
    coalesce(F.col("prod.Category"), F.lit("UNCATEGORIZED")).alias("Category"),
    coalesce(F.col("prod.UnitPrice"), F.lit(0)).alias("UnitPrice")
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Stage 3: Calcular Totales

# COMMAND ----------

# DataStage Stage: Transformer_Calculations
# Derivation: TotalAmount = Quantity * UnitPrice

df_final = df_enriched.withColumn(
    "TotalAmount",
    F.col("Quantity") * F.col("UnitPrice")
)

display(df_final.limit(10))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Escribir Resultado

# COMMAND ----------

df_final.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save("/mnt/processed/orders_enriched")

print("✅ Proceso completado exitosamente")
```

---

## Patrón 3: Aggregation con Group By

### DataStage Job Structure
```
Input → Transformer (filtros) → Aggregator → Sort → Output
```

### Ejemplo: Reporte de Ventas por Cliente

```python
# COMMAND ----------
# MAGIC %md
# MAGIC ## Reporte de Ventas por Cliente
# MAGIC Migrado desde DataStage Job: Customer_Sales_Report

# COMMAND ----------

# Read transactions
df_transactions = spark.read.format("delta") \
    .load("/mnt/processed/transactions")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Stage 1: Filtros (Transformer)

# COMMAND ----------

# DataStage Stage: Transformer_Filter
# Constraints:
#   - TransactionDate >= FirstDayOfYear
#   - Status = "COMPLETED"

from datetime import datetime

year_start = datetime(2024, 1, 1)

df_filtered = df_transactions.filter(
    (F.col("TransactionDate") >= F.lit(year_start)) &
    (F.col("Status") == "COMPLETED")
)

print(f"Transacciones filtradas: {df_filtered.count()}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Stage 2: Aggregation (Aggregator)

# COMMAND ----------

# DataStage Stage: Aggregator_CustomerSales
# Group by: CustomerID, CustomerName
# Calculations:
#   - TotalAmount = SUM(Amount)
#   - TransactionCount = COUNT()
#   - AvgAmount = AVG(Amount)
#   - MinAmount = MIN(Amount)
#   - MaxAmount = MAX(Amount)
#   - FirstTransaction = MIN(TransactionDate)
#   - LastTransaction = MAX(TransactionDate)

df_aggregated = df_filtered.groupBy("CustomerID", "CustomerName").agg(
    F.sum("Amount").alias("TotalAmount"),
    F.count("*").alias("TransactionCount"),
    F.avg("Amount").alias("AvgAmount"),
    F.min("Amount").alias("MinAmount"),
    F.max("Amount").alias("MaxAmount"),
    F.min("TransactionDate").alias("FirstTransaction"),
    F.max("TransactionDate").alias("LastTransaction")
)

# Calcular diferencia de días entre primera y última transacción
df_aggregated = df_aggregated.withColumn(
    "DaysBetweenTransactions",
    F.datediff(F.col("LastTransaction"), F.col("FirstTransaction"))
)

display(df_aggregated.limit(10))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Stage 3: Sort (Sort)

# COMMAND ----------

# DataStage Stage: Sort_ByAmount
# Sort Keys: TotalAmount DESC

df_sorted = df_aggregated.orderBy(F.desc("TotalAmount"))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Write Output

# COMMAND ----------

df_sorted.write.format("delta") \
    .mode("overwrite") \
    .save("/mnt/reports/customer_sales_summary")

# También exportar a CSV para reporting
df_sorted.coalesce(1).write.format("csv") \
    .option("header", "true") \
    .mode("overwrite") \
    .save("/mnt/reports/customer_sales_summary_csv")

print(f"✅ Reporte generado con {df_sorted.count()} clientes")
```

---

## Patrón 4: Slowly Changing Dimension (SCD) Type 2

### DataStage Job Structure
```
Source → Change Capture → Transformer → Update Dimension Table
```

### Migración con Delta Lake Merge

```python
# COMMAND ----------
# MAGIC %md
# MAGIC ## SCD Type 2 Dimension Load
# MAGIC Migrado desde DataStage Job: DimCustomer_SCD2

# COMMAND ----------

from delta.tables import DeltaTable
from datetime import datetime

# COMMAND ----------
# MAGIC %md
# MAGIC ## Read Source Data

# COMMAND ----------

# Nueva data del sistema fuente
df_source = spark.read.format("jdbc") \
    .option("url", jdbc_url) \
    .option("dbtable", "source_customers") \
    .load()

# Agregar columnas de auditoría
current_timestamp = datetime.now()
df_source = df_source \
    .withColumn("EffectiveDate", F.lit(current_timestamp)) \
    .withColumn("EndDate", F.lit(None).cast("timestamp")) \
    .withColumn("IsCurrent", F.lit(True))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Stage 1: Detectar Cambios (Change Capture)

# COMMAND ----------

# DataStage Stage: Change_Capture
# Compare Key: CustomerID
# Compare Columns: Name, Address, Phone, Email

target_path = "/mnt/dimensions/dim_customer"

if DeltaTable.isDeltaTable(spark, target_path):
    dim_customer = DeltaTable.forPath(spark, target_path)
    
    # Obtener registros actuales
    df_current = spark.read.format("delta").load(target_path) \
        .filter(F.col("IsCurrent") == True)
    
    # Detectar cambios (excluding keys)
    compare_cols = ["Name", "Address", "Phone", "Email"]
    
    # Join para identificar cambios
    df_changes = df_source.alias("src").join(
        df_current.alias("curr"),
        on="CustomerID",
        how="left"
    ).filter(
        # Nuevos registros (no existen en dimensión)
        F.col("curr.CustomerID").isNull() |
        # Registros con cambios en columnas comparadas
        (
            (F.col("src.Name") != F.col("curr.Name")) |
            (F.col("src.Address") != F.col("curr.Address")) |
            (F.col("src.Phone") != F.col("curr.Phone")) |
            (F.col("src.Email") != F.col("curr.Email"))
        )
    ).select("src.*")
    
    print(f"Registros con cambios detectados: {df_changes.count()}")
    
else:
    # Primera carga - todos son nuevos
    df_changes = df_source
    print("Primera carga de dimensión")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Stage 2: Update Dimension (SCD Type 2 Logic)

# COMMAND ----------

# DataStage Stage: Transformer_SCD2
# Logic:
#   1. Cerrar registros actuales (set IsCurrent=False, EndDate=now)
#   2. Insertar nuevas versiones (IsCurrent=True, EffectiveDate=now)

if DeltaTable.isDeltaTable(spark, target_path):
    # Paso 1: Cerrar registros antiguos
    dim_customer.alias("target").merge(
        df_changes.alias("source"),
        "target.CustomerID = source.CustomerID AND target.IsCurrent = True"
    ).whenMatchedUpdate(set={
        "IsCurrent": F.lit(False),
        "EndDate": F.lit(current_timestamp)
    }).execute()
    
    # Paso 2: Insertar nuevas versiones
    df_changes.write.format("delta").mode("append").save(target_path)
    
else:
    # Primera carga
    df_changes.write.format("delta").mode("overwrite").save(target_path)

print("✅ Dimensión actualizada exitosamente")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Validación

# COMMAND ----------

# Verificar que cada CustomerID tenga solo un registro actual
validation = spark.read.format("delta").load(target_path) \
    .filter(F.col("IsCurrent") == True) \
    .groupBy("CustomerID").count() \
    .filter(F.col("count") > 1)

if validation.count() > 0:
    print("⚠️ WARNING: Múltiples registros actuales detectados")
    display(validation)
else:
    print("✅ Validación exitosa: Un solo registro actual por CustomerID")

# Estadísticas
stats = spark.read.format("delta").load(target_path).groupBy("IsCurrent").count()
display(stats)
```

---

## Patrón 5: Error Handling y Reject Links

### DataStage Job Structure
```
Input → Transformer → (good data) → Output
              ↓
         (reject link)
              ↓
         Error Output
```

### Migración PySpark

```python
# COMMAND ----------
# MAGIC %md
# MAGIC ## Data Quality Pipeline con Error Handling

# COMMAND ----------

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Stage 1: Read Input

# COMMAND ----------

df_raw = spark.read.format("csv") \
    .option("header", "true") \
    .option("mode", "PERMISSIVE") \
    .option("columnNameOfCorruptRecord", "_corrupt_record") \
    .load("/mnt/raw/transactions.csv")

total_records = df_raw.count()
logger.info(f"Total registros leídos: {total_records}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Stage 2: Data Quality Checks (Transformer con Constraints)

# COMMAND ----------

# DataStage Stage: Transformer_Validation
# Constraints (reject si fallan):
#   - Amount IS NOT NULL
#   - Amount > 0
#   - TransactionDate IS NOT NULL AND IS VALID DATE
#   - CustomerID IS NOT NULL

# Definir reglas de validación
df_with_validation = df_raw \
    .withColumn("is_valid_amount", 
        F.col("Amount").isNotNull() & (F.col("Amount") > 0)
    ) \
    .withColumn("is_valid_date",
        F.col("TransactionDate").isNotNull() & 
        (F.to_date("TransactionDate").isNotNull())
    ) \
    .withColumn("is_valid_customer",
        F.col("CustomerID").isNotNull()
    ) \
    .withColumn("is_valid_record",
        F.col("is_valid_amount") & 
        F.col("is_valid_date") & 
        F.col("is_valid_customer") &
        F.col("_corrupt_record").isNull()
    )

# Separar buenos y malos registros (DataStage reject link)
df_good = df_with_validation.filter(F.col("is_valid_record") == True) \
    .drop("is_valid_amount", "is_valid_date", "is_valid_customer", "is_valid_record", "_corrupt_record")

df_bad = df_with_validation.filter(F.col("is_valid_record") == False)

good_count = df_good.count()
bad_count = df_bad.count()

logger.info(f"Registros válidos: {good_count}")
logger.info(f"Registros rechazados: {bad_count}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Stage 3: Write Good Records (Output)

# COMMAND ----------

# DataStage Stage: Sequential_File_Output
df_good.write.format("delta") \
    .mode("append") \
    .save("/mnt/processed/transactions")

logger.info("✅ Registros válidos escritos exitosamente")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Stage 4: Write Reject Records (Error Output)

# COMMAND ----------

# DataStage Stage: Sequential_File_Rejects
# Agregar metadata de error
df_bad_with_error = df_bad \
    .withColumn("error_timestamp", F.current_timestamp()) \
    .withColumn("error_reason",
        F.when(~F.col("is_valid_amount"), F.lit("Invalid Amount"))
         .when(~F.col("is_valid_date"), F.lit("Invalid Date"))
         .when(~F.col("is_valid_customer"), F.lit("Missing Customer ID"))
         .when(F.col("_corrupt_record").isNotNull(), F.lit("Corrupt Record"))
         .otherwise(F.lit("Unknown Error"))
    )

df_bad_with_error.write.format("delta") \
    .mode("append") \
    .save("/mnt/errors/transactions_rejects")

logger.info(f"⚠️ {bad_count} registros rechazados escritos en error table")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

summary = {
    "total_records": total_records,
    "valid_records": good_count,
    "rejected_records": bad_count,
    "success_rate": f"{(good_count/total_records)*100:.2f}%"
}

print("=" * 50)
print("RESUMEN DE EJECUCIÓN")
print("=" * 50)
for key, value in summary.items():
    print(f"{key}: {value}")
print("=" * 50)

# Alertar si tasa de rechazo es alta
if bad_count / total_records > 0.05:  # > 5%
    logger.warning(f"⚠️ Alta tasa de rechazo: {bad_count/total_records*100:.2f}%")
```

---

## Patrón 6: Procesamiento Complejo con Stage Variables

### DataStage: Transformer con Stage Variables

```basic
' DataStage Logic
' Stage Variables:
'   PrevCustomerID (initial: "") 
'   CustomerRecordCount (initial: 0)
'
' Derivations:
'   If CustomerID = PrevCustomerID Then
'     CustomerRecordCount = CustomerRecordCount + 1
'   Else
'     Begin
'       PrevCustomerID = CustomerID
'       CustomerRecordCount = 1
'     End
```

### Migración PySpark usando Window Functions

```python
# COMMAND ----------
# MAGIC %md
# MAGIC ## Complex Transformation con Lógica de Secuencia

# COMMAND ----------

from pyspark.sql.window import Window

# DataStage Stage: Transformer_WithStageVars
# Stage Variables simuladas con Window Functions

# Definir ventana para particionar por CustomerID y ordenar
window_spec = Window.partitionBy("CustomerID").orderBy("TransactionDate")

df_with_sequence = df.withColumn(
    "CustomerRecordCount",
    F.row_number().over(window_spec)
).withColumn(
    "IsFirstTransaction",
    F.when(F.col("CustomerRecordCount") == 1, True).otherwise(False)
).withColumn(
    "RunningTotal",
    F.sum("Amount").over(window_spec.rowsBetween(Window.unboundedPreceding, Window.currentRow))
)

display(df_with_sequence)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Detectar Cambios de Grupo (Change Detection)

# COMMAND ----------

# DataStage: Stage variable para detectar cambio de CustomerID
# PySpark: Usar lag function

window_all = Window.orderBy("TransactionDate")

df_with_changes = df.withColumn(
    "PrevCustomerID",
    F.lag ("CustomerID", 1).over(window_all)
).withColumn(
    "IsCustomerChange",
    F.when(
        (F.col("CustomerID") != F.col("PrevCustomerID")) | 
        F.col("PrevCustomerID").isNull(),
        True
    ).otherwise(False)
)

display(df_with_changes)
```

---

## Tips de Migración

### 1. Preservar Logging y Metadata
```python
# Agregar metadata de procesamiento
df_with_metadata = df \
    .withColumn("process_timestamp", F.current_timestamp()) \
    .withColumn("source_file", F.input_file_name()) \
    .withColumn("job_name", F.lit("customer_etl")) \
    .withColumn("job_run_id", F.lit(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()))
```

### 2. Validaciones de Calidad
```python
# Validar counts antes y después
assert df_output.count() == expected_count, f"Count mismatch: expected {expected_count}, got {df_output.count()}"

# Checksum validation
checksum_before = df_input.select(F.sum(F.hash(*df_input.columns))).collect()[0][0]
checksum_after = df_output.select(F.sum(F.hash(*df_output.columns))).collect()[0][0]
```

### 3. Performance Optimization
```python
# Repartition antes de writes costosos
df.repartition(200).write...

# Persist DataFrames que se usan múltiples veces
df_cached = df.cache()
df_cached.count()  # Trigger caching

# Cleanup después de uso
df_cached.unpersist()
```

### 4. Manejo de Parámetros DataStage
```python
# DataStage Job Parameters → Databricks Widgets
dbutils.widgets.text("DS_INPUT_PATH", "")
dbutils.widgets.text("DS_OUTPUT_PATH", "")
dbutils.widgets.text("DS_RUN_DATE", "")
dbutils.widgets.dropdown("DS_LOAD_TYPE", "FULL", ["FULL", "INCREMENTAL"])

# Usar en código
input_path = dbutils.widgets.get("DS_INPUT_PATH")
load_type = dbutils.widgets.get("DS_LOAD_TYPE")
```
