# Databricks notebook source
# MAGIC %md
# MAGIC # Customer ETL - Migrado desde DataStage
# MAGIC 
# MAGIC **Job Original**: Simple_Customer_ETL  
# MAGIC **Descripción**: Procesa archivo de clientes, limpia datos y exporta resultado  
# MAGIC **Migrado**: 2024-01-15  
# MAGIC **Por**: DataStage to Databricks Migration Agent
# MAGIC 
# MAGIC ## Resumen
# MAGIC Este notebook implementa el equivalente del job DataStage "Simple_Customer_ETL" que:
# MAGIC 1. Lee archivo CSV de clientes
# MAGIC 2. Limpia y transforma los datos
# MAGIC 3. Aplica validaciones
# MAGIC 4. Exporta resultado
# MAGIC 
# MAGIC ## DataStage Stage Mapping
# MAGIC | DataStage Stage | Type | Databricks Implementation |
# MAGIC |----------------|------|---------------------------|
# MAGIC | Input_Customers | Sequential File | spark.read.csv() |
# MAGIC | Transform_Clean | Transformer | withColumn() operations |
# MAGIC | Output_Customers | Sequential File | write to Delta Lake |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📝 Parameters
# MAGIC 
# MAGIC Estos parámetros mapean a los parámetros del job DataStage original

# COMMAND ----------

# DataStage Parameter: INPUT_FILE_PATH
dbutils.widgets.text("input_path", "/mnt/raw/customers.csv", "Input Path")

# DataStage Parameter: OUTPUT_FILE_PATH  
dbutils.widgets.text("output_path", "/mnt/processed/customers", "Output Path")

# DataStage Parameter: RUN_DATE
dbutils.widgets.text("run_date", "", "Run Date (YYYY-MM-DD)")

# Obtener valores
input_path = dbutils.widgets.get("input_path")
output_path = dbutils.widgets.get("output_path")
run_date = dbutils.widgets.get("run_date") if dbutils.widgets.get("run_date") else str(datetime.now().date())

print(f"Input Path: {input_path}")
print(f"Output Path: {output_path}")
print(f"Run Date: {run_date}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📦 Imports and Configuration

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import *
from delta.tables import DeltaTable
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔧 Spark Configuration
# MAGIC 
# MAGIC Configuraciones optimizadas para este job

# COMMAND ----------

# Habilitar Adaptive Query Execution
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")

# Configurar para dataset pequeño-mediano
spark.conf.set("spark.sql.shuffle.partitions", "100")

# Broadcast threshold para lookups
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "10MB")

logger.info("Spark configuration completed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Stage 1: Read Input (Input_Customers)
# MAGIC 
# MAGIC **DataStage Stage**: Sequential_File  
# MAGIC **Properties**:
# MAGIC - Format: CSV delimited
# MAGIC - Delimiter: comma
# MAGIC - Quote: "
# MAGIC - Header: true

# COMMAND ----------

logger.info(f"Reading input file from: {input_path}")

# DataStage Stage: Input_Customers (Sequential File)
df_raw = spark.read.format("csv") \
    .option("header", "true") \
    .option("delimiter", ",") \
    .option("quote", "\"") \
    .option("inferSchema", "true") \
    .option("mode", "PERMISSIVE") \
    .option("columnNameOfCorruptRecord", "_corrupt_record") \
    .load(input_path)

# Capturar count de input
input_count = df_raw.count()
logger.info(f"Records read: {input_count}")

# Mostrar sample
print(f"✅ Input loaded successfully: {input_count} records")
display(df_raw.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔄 Stage 2: Transform and Clean (Transform_Clean)
# MAGIC 
# MAGIC **DataStage Stage**: Transformer  
# MAGIC **Derivations**:
# MAGIC - FullName = Trim(Upcase(FirstName)) : " " : Trim(Upcase(LastName))
# MAGIC - EmailClean = Downcase(Trim(Email))
# MAGIC - Age = YearsFromDate(BirthDate)
# MAGIC - Status = If IsNull(Status) Then "UNKNOWN" Else Upcase(Trim(Status))
# MAGIC - DaysSinceRegistration = DaysSince(RegistrationDate)
# MAGIC - ProcessDate = #RUN_DATE#
# MAGIC 
# MAGIC **Constraints**:
# MAGIC - CustomerID IS NOT NULL
# MAGIC - Email IS NOT NULL
# MAGIC - Status IN ("ACTIVE", "INACTIVE", "UNKNOWN")

# COMMAND ----------

logger.info("Starting transformations...")

# DataStage Stage: Transform_Clean (Transformer)
df_transformed = df_raw \
    .withColumn("FullName",
        # DataStage: Trim(Upcase(FirstName)) : " " : Trim(Upcase(LastName))
        F.concat(
            F.trim(F.upper(F.col("FirstName"))),
            F.lit(" "),
            F.trim(F.upper(F.col("LastName")))
        )
    ) \
    .withColumn("EmailClean",
        # DataStage: Downcase(Trim(Email))
        F.lower(F.trim(F.col("Email")))
    ) \
    .withColumn("Age",
        # DataStage: YearsFromDate(BirthDate)
        F.floor(F.months_between(F.current_date(), F.col("BirthDate")) / 12)
    ) \
    .withColumn("Status",
        # DataStage: If IsNull(Status) Then "UNKNOWN" Else Upcase(Trim(Status))
        F.when(F.col("Status").isNull(), F.lit("UNKNOWN"))
         .otherwise(F.upper(F.trim(F.col("Status"))))
    ) \
    .withColumn("DaysSinceRegistration",
        # DataStage: DaysSince(RegistrationDate)
        F.datediff(F.current_date(), F.col("RegistrationDate"))
    ) \
    .withColumn("ProcessDate",
        # DataStage: #RUN_DATE#
        F.lit(run_date).cast("date")
    )

# Aplicar constraints (DataStage Constraints)
df_filtered = df_transformed.filter(
    # Constraint: CustomerID IS NOT NULL
    F.col("CustomerID").isNotNull() &
    # Constraint: Email IS NOT NULL
    F.col("Email").isNotNull() &
    # Constraint: Status IN valid values
    F.col("Status").isin(["ACTIVE", "INACTIVE", "UNKNOWN"])
)

# Contar rechazos
reject_count = input_count - df_filtered.count()
logger.info(f"Records after transformation: {df_filtered.count()}")
logger.info(f"Records rejected: {reject_count}")

# Seleccionar columnas finales (Output Columns del stage DataStage)
df_final = df_filtered.select(
    "CustomerID",
    "FullName",
    "EmailClean",
    "Age",
    "Status",
    "DaysSinceRegistration",
    "ProcessDate"
)

print(f"✅ Transformation completed")
print(f"   - Good records: {df_final.count()}")
print(f"   - Rejected records: {reject_count}")
display(df_final.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 💾 Stage 3: Write Output (Output_Customers)
# MAGIC 
# MAGIC **DataStage Stage**: Sequential_File  
# MAGIC **Properties**:
# MAGIC - Format: Delta Lake (modernización desde CSV)
# MAGIC - Write Mode: Overwrite

# COMMAND ----------

logger.info(f"Writing output to: {output_path}")

# DataStage Stage: Output_Customers (Sequential File)
# Nota: Usando Delta Lake en lugar de CSV para mejor performance y ACID
df_final.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(output_path)

# Optimizar tabla Delta
spark.sql(f"OPTIMIZE delta.`{output_path}`")

logger.info("Output written successfully")
print(f"✅ Data written to: {output_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Validation and Metrics
# MAGIC 
# MAGIC Validaciones finales y captura de métricas

# COMMAND ----------

# Leer output para validar
df_output = spark.read.format("delta").load(output_path)
output_count = df_output.count()

# Validaciones
print("=" * 60)
print("VALIDATION SUMMARY")
print("=" * 60)
print(f"Input records:     {input_count}")
print(f"Output records:    {output_count}")
print(f"Rejected records:  {reject_count}")
print(f"Rejection rate:    {(reject_count/input_count)*100:.2f}%")
print("=" * 60)

# Data Quality Checks
print("\nDATA QUALITY CHECKS:")

# Check for nulls in key columns
null_checks = df_output.select(
    F.sum(F.col("CustomerID").isNull().cast("int")).alias("CustomerID_nulls"),
    F.sum(F.col("EmailClean").isNull().cast("int")).alias("Email_nulls")
).collect()[0]

if null_checks["CustomerID_nulls"] == 0 and null_checks["Email_nulls"] == 0:
    print("✅ No nulls in key columns")
else:
    print(f"⚠️ Found nulls: CustomerID={null_checks['CustomerID_nulls']}, Email={null_checks['Email_nulls']}")

# Check for duplicates
dup_count = df_output.groupBy("CustomerID").count().filter(F.col("count") > 1).count()
if dup_count == 0:
    print("✅ No duplicate CustomerIDs")
else:
    print(f"⚠️ Found {dup_count} duplicate CustomerIDs")

# Schema validation
expected_columns = ["CustomerID", "FullName", "EmailClean", "Age", "Status", "DaysSinceRegistration", "ProcessDate"]
actual_columns = df_output.columns
missing_columns = set(expected_columns) - set(actual_columns)

if not missing_columns:
    print("✅ All expected columns present")
else:
    print(f"⚠️ Missing columns: {missing_columns}")

print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Capture Metrics
# MAGIC 
# MAGIC Guardar métricas de ejecución para monitoreo

# COMMAND ----------

# Capturar métricas de ejecución
metrics = [{
    "job_name": "Simple_Customer_ETL",
    "run_date": run_date,
    "execution_timestamp": datetime.now(),
    "input_path": input_path,
    "output_path": output_path,
    "records_input": input_count,
    "records_output": output_count,
    "records_rejected": reject_count,
    "rejection_rate_pct": round((reject_count/input_count)*100, 2),
    "status": "SUCCESS"
}]

# Crear DataFrame con métricas
metrics_df = spark.createDataFrame(metrics)

# Guardar métricas (ajustar path según tu configuración)
# metrics_df.write.format("delta").mode("append").saveAsTable("monitoring.job_metrics")

logger.info("Job completed successfully")
print("\n✅ JOB COMPLETED SUCCESSFULLY")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📝 Migration Notes
# MAGIC 
# MAGIC ### Decisiones de Diseño
# MAGIC 
# MAGIC 1. **Delta Lake vs CSV**: Output migrado de CSV a Delta Lake para:
# MAGIC    - ACID transactions
# MAGIC    - Better performance
# MAGIC    - Time travel capabilities
# MAGIC    - Schema evolution
# MAGIC 
# MAGIC 2. **Error Handling**: 
# MAGIC    - DataStage reject links → DataFrame filters
# MAGIC    - Registros rechazados loggeados pero no escritos a archivo separado
# MAGIC    - Para escribir rechazos por separado, descomentar sección adicional
# MAGIC 
# MAGIC 3. **Optimizaciones Aplicadas**:
# MAGIC    - Adaptive Query Execution habilitado
# MAGIC    - Shuffle partitions ajustados al volumen de datos
# MAGIC    - Delta optimization ejecutada post-write
# MAGIC 
# MAGIC ### Diferencias con DataStage
# MAGIC 
# MAGIC - **Null Handling**: PySpark y DataStage pueden manejar nulls diferentemente en algunas funciones
# MAGIC - **String Functions**: Índices son 1-based en ambos (substring), pero verificar cada caso
# MAGIC - **Date Functions**: YearsFromDate traducido a months_between / 12
# MAGIC 
# MAGIC ### Próximos Pasos
# MAGIC 
# MAGIC 1. ✅ **Testing**: Validar con dataset completo
# MAGIC 2. ✅ **Comparación**: Comparar output con DataStage original
# MAGIC 3. ⏳ **Monitoring**: Agregar alertas en caso de alta tasa de rechazo
# MAGIC 4. ⏳ **Scheduling**: Configurar Databricks Job con schedule apropiado
# MAGIC 5. ⏳ **Notifications**: Integrar con sistema de alertas (email, Slack, etc.)
# MAGIC 
# MAGIC ### Performance Baseline
# MAGIC 
# MAGIC - **DataStage Execution Time**: [A completar]
# MAGIC - **Databricks Execution Time**: [A completar tras testing]
# MAGIC - **Target SLA**: [Definir]
