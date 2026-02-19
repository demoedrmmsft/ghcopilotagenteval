# Databricks notebook source
# MAGIC %md
# MAGIC # Simple Customer ETL - Migrated from DataStage
# MAGIC 
# MAGIC **Original Job**: Simple_Customer_ETL.dsx
# MAGIC 
# MAGIC **Description**: Procesa archivo de clientes - limpia y exporta resultado
# MAGIC 
# MAGIC **Job Version**: 1.0
# MAGIC 
# MAGIC **Migration Date**: 2026-02-17
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC ## Job Flow
# MAGIC 
# MAGIC ```
# MAGIC Input_Customers (CSV)
# MAGIC         ↓
# MAGIC   Transform_Clean
# MAGIC    - FullName: TRIM + UPPER concatenation
# MAGIC    - EmailClean: lowercase + trim
# MAGIC    - Age: Years from birth date
# MAGIC    - Status: Handle nulls
# MAGIC    - DaysSinceRegistration: Date difference
# MAGIC         ↓
# MAGIC Output_Customers (Delta Lake - MODERNIZED)
# MAGIC ```
# MAGIC 
# MAGIC ## Data Quality Rules
# MAGIC - ✅ CustomerID cannot be NULL
# MAGIC - ✅ Email cannot be NULL
# MAGIC - ✅ Status must be: ACTIVE, INACTIVE, or UNKNOWN

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Parameters Setup
# MAGIC Configure job parameters using Databricks widgets (equivalent to DataStage parameters)

# COMMAND ----------

# Create widgets for job parameters
dbutils.widgets.text("INPUT_FILE_PATH", "/data/input/customers.csv", "Input File Path")
dbutils.widgets.text("OUTPUT_FILE_PATH", "/data/output/customers_clean", "Output Path (Delta Lake)")
dbutils.widgets.text("RUN_DATE", "", "Run Date (YYYY-MM-DD, empty = today)")

# Get parameter values
INPUT_FILE_PATH = dbutils.widgets.get("INPUT_FILE_PATH")
OUTPUT_FILE_PATH = dbutils.widgets.get("OUTPUT_FILE_PATH")
RUN_DATE = dbutils.widgets.get("RUN_DATE")

# If RUN_DATE is empty, use today's date (equivalent to #{CurrentDate})
if not RUN_DATE:
    from datetime import date
    RUN_DATE = date.today().isoformat()

print("=" * 60)
print("📋 JOB PARAMETERS")
print("=" * 60)
print(f"📥 Input Path:  {INPUT_FILE_PATH}")
print(f"📤 Output Path: {OUTPUT_FILE_PATH}")
print(f"📅 Run Date:    {RUN_DATE}")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Imports and Spark Configuration

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import *
from delta.tables import DeltaTable
from datetime import datetime

# Enable Adaptive Query Execution for better performance
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")

# Set shuffle partitions based on data size (adjust as needed)
spark.conf.set("spark.sql.shuffle.partitions", "8")

# Enable Delta Lake optimizations
spark.conf.set("spark.databricks.delta.optimizeWrite.enabled", "true")
spark.conf.set("spark.databricks.delta.autoCompact.enabled", "true")

print("✅ Spark configuration applied")
print(f"   - Adaptive Query Execution: ENABLED")
print(f"   - Delta Lake Auto-Optimize: ENABLED")
print(f"   - Shuffle Partitions: 8")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Input Stage - Read Customer Data
# MAGIC **DataStage Stage**: `Input_Customers` (Sequential_File - Source)
# MAGIC 
# MAGIC **Columns**:
# MAGIC - CustomerID (INTEGER, NOT NULL)
# MAGIC - FirstName (VARCHAR 50)
# MAGIC - LastName (VARCHAR 50)
# MAGIC - Email (VARCHAR 100)
# MAGIC - BirthDate (DATE)
# MAGIC - Status (VARCHAR 20)
# MAGIC - RegistrationDate (DATE)

# COMMAND ----------

# Read CSV file with schema inference and header
# Equivalent to DataStage Sequential_File stage with Delimited format
df_input = (spark.read
    .format("csv")
    .option("header", "true")           # FirstLineIsColumnNames = True
    .option("inferSchema", "true")      # Auto-detect data types
    .option("delimiter", ",")           # ColumnDelimiter = ","
    .option("quote", '"')               # QuoteCharacter = "
    .option("escape", '"')
    .option("mode", "PERMISSIVE")       # Handle malformed records
    .option("nullValue", "")
    .option("dateFormat", "yyyy-MM-dd")
    .load(INPUT_FILE_PATH)
)

# Validate expected columns exist
expected_columns = ["CustomerID", "FirstName", "LastName", "Email", "BirthDate", "Status", "RegistrationDate"]
actual_columns = df_input.columns
missing_columns = set(expected_columns) - set(actual_columns)

if missing_columns:
    raise ValueError(f"❌ Missing required columns: {missing_columns}")

input_count = df_input.count()

print(f"✅ Input data loaded successfully")
print(f"📊 Total records: {input_count:,}")
print(f"\n📋 Input Schema:")
df_input.printSchema()

# Display sample data
print("\n🔍 Sample Input Data (first 5 rows):")
display(df_input.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Transform Stage - Data Cleaning and Derivations
# MAGIC **DataStage Stage**: `Transform_Clean` (Transformer)
# MAGIC 
# MAGIC ### Derivations Applied:
# MAGIC 
# MAGIC | Column | DataStage Expression | PySpark Translation |
# MAGIC |--------|---------------------|---------------------|
# MAGIC | CustomerID | `CustomerID` | Pass-through |
# MAGIC | FullName | `Trim(Upcase(FirstName)) : " " : Trim(Upcase(LastName))` | `F.concat(F.trim(F.upper(...)))` |
# MAGIC | EmailClean | `Downcase(Trim(Email))` | `F.lower(F.trim(...))` |
# MAGIC | Age | `YearsFromDate(BirthDate)` | `F.floor(F.months_between(...) / 12)` |
# MAGIC | Status | `If IsNull(Status) Then "UNKNOWN" Else Upcase(Trim(Status))` | `F.when(...isNull()...).otherwise(...)` |
# MAGIC | DaysSinceRegistration | `DaysSince(RegistrationDate)` | `F.datediff(...)` |
# MAGIC | ProcessDate | `#RUN_DATE#` | `F.lit(RUN_DATE)` |

# COMMAND ----------

# Apply all transformations
df_transformed = (df_input
    
    # Derivation 1: CustomerID (pass-through)
    .withColumn("CustomerID", F.col("CustomerID"))
    
    # Derivation 2: FullName
    # DataStage: Trim(Upcase(FirstName)) : " " : Trim(Upcase(LastName))
    # Description: Concatenar y limpiar nombre completo
    .withColumn("FullName", 
        F.concat(
            F.trim(F.upper(F.col("FirstName"))),
            F.lit(" "),
            F.trim(F.upper(F.col("LastName")))
        )
    )
    
    # Derivation 3: EmailClean
    # DataStage: Downcase(Trim(Email))
    # Description: Normalizar email a minúsculas
    .withColumn("EmailClean", 
        F.lower(F.trim(F.col("Email")))
    )
    
    # Derivation 4: Age
    # DataStage: YearsFromDate(BirthDate)
    # Description: Calcular edad desde fecha de nacimiento
    # Note: months_between returns fractional months, divide by 12 and floor for years
    .withColumn("Age", 
        F.floor(F.months_between(F.current_date(), F.col("BirthDate")) / 12).cast(IntegerType())
    )
    
    # Derivation 5: Status
    # DataStage: If IsNull(Status) Then "UNKNOWN" Else Upcase(Trim(Status))
    # Description: Manejar status nulo
    .withColumn("Status", 
        F.when(F.col("Status").isNull(), F.lit("UNKNOWN"))
         .otherwise(F.upper(F.trim(F.col("Status"))))
    )
    
    # Derivation 6: DaysSinceRegistration
    # DataStage: DaysSince(RegistrationDate)
    # Description: Días desde registro
    .withColumn("DaysSinceRegistration", 
        F.datediff(F.current_date(), F.col("RegistrationDate"))
    )
    
    # Derivation 7: ProcessDate
    # DataStage: #RUN_DATE#
    # Description: Fecha de procesamiento
    .withColumn("ProcessDate", 
        F.lit(RUN_DATE).cast(DateType())
    )
)

print(f"✅ Transformations applied successfully")
print(f"📊 Transformed records: {df_transformed.count():,}")
print(f"\n📋 Transformed Schema:")
df_transformed.printSchema()

# Display sample transformed data
print("\n🔍 Sample Transformed Data (first 5 rows):")
display(df_transformed.select(
    "CustomerID", "FullName", "EmailClean", "Age", 
    "Status", "DaysSinceRegistration", "ProcessDate"
).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Data Quality Constraints
# MAGIC **DataStage Constraints**:
# MAGIC 1. `NOT(IsNull(CustomerID))` - CustomerID no puede ser nulo
# MAGIC 2. `NOT(IsNull(Email))` - Email no puede ser nulo
# MAGIC 3. `Status = "ACTIVE" OR Status = "INACTIVE" OR Status = "UNKNOWN"` - Status debe ser valor válido

# COMMAND ----------

# Count records before constraints
count_before_validation = df_transformed.count()

# Track rejected records for each constraint
rejected_customerid = df_transformed.filter(F.col("CustomerID").isNull()).count()
rejected_email = df_transformed.filter(F.col("Email").isNull()).count()
rejected_status = df_transformed.filter(~F.col("Status").isin(["ACTIVE", "INACTIVE", "UNKNOWN"])).count()

# Apply all constraints as filters
df_validated = (df_transformed
    # Constraint 1: NOT(IsNull(CustomerID))
    .filter(F.col("CustomerID").isNotNull())
    
    # Constraint 2: NOT(IsNull(Email))
    .filter(F.col("Email").isNotNull())
    
    # Constraint 3: Status must be valid value
    .filter(F.col("Status").isin(["ACTIVE", "INACTIVE", "UNKNOWN"]))
)

count_after_validation = df_validated.count()
total_rejected = count_before_validation - count_after_validation

print("=" * 60)
print("🔍 DATA QUALITY VALIDATION RESULTS")
print("=" * 60)
print(f"📊 Records before validation: {count_before_validation:,}")
print(f"✅ Records after validation:  {count_after_validation:,}")
print(f"❌ Total rejected records:    {total_rejected:,}")
print("\n📋 Rejection Breakdown:")
print(f"   - CustomerID is NULL:      {rejected_customerid:,}")
print(f"   - Email is NULL:           {rejected_email:,}")
print(f"   - Invalid Status value:    {rejected_status:,}")

if total_rejected > 0:
    rejection_rate = (total_rejected / count_before_validation) * 100
    print(f"\n⚠️  Rejection rate: {rejection_rate:.2f}%")
    if rejection_rate > 5:
        print(f"⚠️  WARNING: High rejection rate detected!")
else:
    print(f"\n✅ No rejections - all data passed validation!")

print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Output Stage - Write to Delta Lake
# MAGIC **DataStage Stage**: `Output_Customers` (Sequential_File - Target)
# MAGIC 
# MAGIC **Modernization Applied**: 
# MAGIC - ❌ Original: CSV file (Sequential_File with Delimited format)
# MAGIC - ✅ Upgraded: Delta Lake format
# MAGIC 
# MAGIC **Benefits of Delta Lake**:
# MAGIC - 🔒 ACID transactions
# MAGIC - ⏰ Time travel (data versioning)
# MAGIC - 📊 Schema enforcement and evolution
# MAGIC - ⚡ Performance optimizations (Z-ORDER, AUTO OPTIMIZE)
# MAGIC - 🔄 Supports MERGE operations for incremental updates

# COMMAND ----------

# Select final output columns in correct order
# Matches DataStage Output_Customers stage column definition
df_output = df_validated.select(
    F.col("CustomerID").cast(IntegerType()),
    F.col("FullName").cast(StringType()),
    F.col("EmailClean").cast(StringType()),
    F.col("Age").cast(IntegerType()),
    F.col("Status").cast(StringType()),
    F.col("DaysSinceRegistration").cast(IntegerType()),
    F.col("ProcessDate").cast(DateType())
)

# Write to Delta Lake (modernized from CSV)
# WriteMode = OVERWRITE in DataStage → mode("overwrite") in PySpark
(df_output
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .option("mergeSchema", "false")
    .save(OUTPUT_FILE_PATH)
)

output_count = df_output.count()

print("=" * 60)
print("✅ DATA WRITTEN TO DELTA LAKE")
print("=" * 60)
print(f"📍 Location:        {OUTPUT_FILE_PATH}")
print(f"📊 Records written: {output_count:,}")
print(f"📁 Format:          Delta Lake (upgraded from CSV)")
print(f"💾 Write mode:      OVERWRITE")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Post-Write Validation and Verification

# COMMAND ----------

# Read back the data to validate write was successful
df_verify = spark.read.format("delta").load(OUTPUT_FILE_PATH)

# Validation metrics
verify_count = df_verify.count()
distinct_customers = df_verify.select("CustomerID").distinct().count()
duplicate_customers = verify_count - distinct_customers

print("=" * 60)
print("🔍 OUTPUT VALIDATION")
print("=" * 60)
print(f"✅ Records verified in output: {verify_count:,}")
print(f"👥 Distinct customers:         {distinct_customers:,}")
print(f"🔄 Duplicate records:          {duplicate_customers:,}")

# Verify counts match
if verify_count == output_count:
    print(f"✅ Count verification PASSED")
else:
    print(f"⚠️  Count mismatch detected!")

# Data quality checks on output
null_checks = {
    "CustomerID": df_verify.filter(F.col("CustomerID").isNull()).count(),
    "EmailClean": df_verify.filter(F.col("EmailClean").isNull()).count(),
    "FullName": df_verify.filter(F.col("FullName").isNull()).count(),
    "Status": df_verify.filter(F.col("Status").isNull()).count()
}

print("\n📋 Output Data Quality Checks:")
all_passed = True
for column, null_count in null_checks.items():
    status_icon = "✅" if null_count == 0 else "❌"
    print(f"{status_icon} {column:20} NULL count: {null_count:,}")
    if null_count > 0:
        all_passed = False

if all_passed:
    print("\n✅ All quality checks PASSED")
else:
    print("\n⚠️  Some quality checks FAILED")

print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Data Profiling and Statistics

# COMMAND ----------

print("=" * 60)
print("📊 STATUS DISTRIBUTION")
print("=" * 60)
df_verify.groupBy("Status").count().orderBy(F.desc("count")).show(truncate=False)

print("=" * 60)
print("📊 AGE STATISTICS")
print("=" * 60)
df_verify.select(
    F.min("Age").alias("Min_Age"),
    F.max("Age").alias("Max_Age"),
    F.round(F.avg("Age"), 2).alias("Avg_Age"),
    F.round(F.stddev("Age"), 2).alias("StdDev_Age"),
    F.expr("percentile(Age, 0.5)").alias("Median_Age")
).show()

print("=" * 60)
print("📊 DAYS SINCE REGISTRATION STATISTICS")
print("=" * 60)
df_verify.select(
    F.min("DaysSinceRegistration").alias("Min_Days"),
    F.max("DaysSinceRegistration").alias("Max_Days"),
    F.round(F.avg("DaysSinceRegistration"), 2).alias("Avg_Days")
).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Optimize Delta Table

# COMMAND ----------

# Optimize the Delta table for better read performance
# This compacts small files and improves query performance
spark.sql(f"OPTIMIZE delta.`{OUTPUT_FILE_PATH}`")

# Collect statistics for the Catalyst optimizer
spark.sql(f"ANALYZE TABLE delta.`{OUTPUT_FILE_PATH}` COMPUTE STATISTICS")

print("=" * 60)
print("✅ DELTA TABLE OPTIMIZATION COMPLETE")
print("=" * 60)
print("🔧 Actions performed:")
print("   - OPTIMIZE: Compacted small files")
print("   - ANALYZE: Collected table statistics")
print("\n💡 For large tables, consider Z-ORDER indexing:")
print(f"   OPTIMIZE delta.`{OUTPUT_FILE_PATH}` ZORDER BY (CustomerID)")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Job Execution Summary

# COMMAND ----------

# Generate comprehensive job execution summary
execution_end_time = datetime.now()

summary = {
    "Job Name": "Simple_Customer_ETL",
    "Original Format": "DataStage DSX",
    "Target Platform": "Azure Databricks",
    "Execution Date": execution_end_time.strftime("%Y-%m-%d %H:%M:%S"),
    "Run Date Parameter": RUN_DATE,
    "Input Path": INPUT_FILE_PATH,
    "Output Path": OUTPUT_FILE_PATH,
    "Output Format": "Delta Lake (upgraded from CSV)",
    "─" * 30: "─" * 30,
    "Input Records": f"{input_count:,}",
    "After Transformation": f"{count_before_validation:,}",
    "After Validation": f"{count_after_validation:,}",
    "Output Records": f"{output_count:,}",
    "Rejected Records": f"{total_rejected:,}",
    "Rejection Rate": f"{(total_rejected / input_count * 100) if input_count > 0 else 0:.2f}%",
    "──" * 30: "──" * 30,
    "Execution Status": "✅ SUCCESS" if all_passed else "⚠️  COMPLETED WITH WARNINGS"
}

print("=" * 70)
print(" " * 20 + "📋 JOB EXECUTION SUMMARY")
print("=" * 70)
for key, value in summary.items():
    if "─" in key:
        print(key + value)
    else:
        print(f"{key:.<35} {value}")
print("=" * 70)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Migration Reference Guide
# MAGIC 
# MAGIC ## DataStage to PySpark Expression Mappings
# MAGIC 
# MAGIC This job used the following expression translations:
# MAGIC 
# MAGIC | # | DataStage Expression | PySpark Equivalent | Notes |
# MAGIC |---|---------------------|-------------------|-------|
# MAGIC | 1 | `Trim(Upcase(FirstName))` | `F.trim(F.upper(F.col("FirstName")))` | Nested function calls |
# MAGIC | 2 | `str1 : " " : str2` | `F.concat(str1, F.lit(" "), str2)` | DataStage ":" is concat |
# MAGIC | 3 | `Downcase(Trim(Email))` | `F.lower(F.trim(F.col("Email")))` | Case conversion |
# MAGIC | 4 | `YearsFromDate(BirthDate)` | `F.floor(F.months_between(F.current_date(), col) / 12)` | Date arithmetic |
# MAGIC | 5 | `If IsNull(x) Then a Else b` | `F.when(F.col("x").isNull(), a).otherwise(b)` | Conditional logic |
# MAGIC | 6 | `DaysSince(RegistrationDate)` | `F.datediff(F.current_date(), col)` | Date difference |
# MAGIC | 7 | `#RUN_DATE#` | `F.lit(RUN_DATE)` | Parameter reference |
# MAGIC | 8 | `NOT(IsNull(col))` | `F.col("col").isNotNull()` | Boolean negation |
# MAGIC | 9 | `Status = "A" OR Status = "B"` | `F.col("Status").isin(["A", "B"])` | Multiple OR conditions |
# MAGIC 
# MAGIC ## Component Mappings
# MAGIC 
# MAGIC | DataStage Component | PySpark Equivalent | Implementation |
# MAGIC |--------------------|-------------------|----------------|
# MAGIC | Sequential_File (Source) | `spark.read.csv()` | Read with schema inference |
# MAGIC | Transformer | `withColumn()` | Chain multiple transformations |
# MAGIC | Constraints | `filter()` | Apply as DataFrame filters |
# MAGIC | Sequential_File (Target) | `write.format("delta")` | **MODERNIZED to Delta Lake** |
# MAGIC 
# MAGIC ## Modernizations Applied
# MAGIC 
# MAGIC ### 1. Delta Lake Instead of CSV
# MAGIC **Why**: ACID transactions, time travel, schema enforcement, better performance
# MAGIC 
# MAGIC ```python
# MAGIC # Before (DataStage - CSV)
# MAGIC Sequential_File → WriteMode: OVERWRITE → CSV file
# MAGIC 
# MAGIC # After (Databricks - Delta Lake)
# MAGIC df.write.format("delta").mode("overwrite").save(path)
# MAGIC ```
# MAGIC 
# MAGIC **Benefits**:
# MAGIC - ✅ ACID compliance
# MAGIC - ✅ Time travel: `spark.read.format("delta").option("versionAsOf", 0).load(path)`
# MAGIC - ✅ Schema evolution support
# MAGIC - ✅ Automatic file compaction
# MAGIC - ✅ Better compression and performance
# MAGIC 
# MAGIC ### 2. Adaptive Query Execution (AQE)
# MAGIC **Why**: Dynamic optimization based on runtime statistics
# MAGIC 
# MAGIC ```python
# MAGIC spark.conf.set("spark.sql.adaptive.enabled", "true")
# MAGIC spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
# MAGIC ```
# MAGIC 
# MAGIC **Benefits**:
# MAGIC - ✅ Automatic partition coalescing
# MAGIC - ✅ Skew join optimization
# MAGIC - ✅ Dynamic query replanning
# MAGIC 
# MAGIC ### 3. Enhanced Data Quality Validation
# MAGIC **Why**: Better visibility into data quality issues
# MAGIC 
# MAGIC - Track rejection counts per constraint
# MAGIC - Calculate rejection rates
# MAGIC - Post-write validation checks
# MAGIC - Statistical profiling
# MAGIC 
# MAGIC ### 4. Performance Optimization
# MAGIC **Why**: Production-ready performance from day one
# MAGIC 
# MAGIC - `OPTIMIZE` command for file compaction
# MAGIC - `ANALYZE` for statistics collection
# MAGIC - Recommendation for Z-ORDER indexing
# MAGIC 
# MAGIC ## Next Steps for Production Deployment
# MAGIC 
# MAGIC ### 1. Environment Configuration
# MAGIC ```python
# MAGIC # Update paths for your environment
# MAGIC INPUT_FILE_PATH = "abfss://container@storage.dfs.core.windows.net/input/customers.csv"
# MAGIC OUTPUT_FILE_PATH = "abfss://container@storage.dfs.core.windows.net/output/customers_clean"
# MAGIC ```
# MAGIC 
# MAGIC ### 2. Schedule as Databricks Job
# MAGIC - Create a job in Databricks workspace
# MAGIC - Set schedule (cron expression)
# MAGIC - Configure alerts for failures
# MAGIC - Set up retry policies
# MAGIC 
# MAGIC ### 3. Incremental Load Strategy (Optional)
# MAGIC If this needs to run incrementally:
# MAGIC 
# MAGIC ```python
# MAGIC # Add watermark column to track last processed date
# MAGIC df_new = df_input.filter(F.col("RegistrationDate") > last_watermark)
# MAGIC 
# MAGIC # Use MERGE instead of OVERWRITE
# MAGIC DeltaTable.forPath(spark, OUTPUT_PATH).alias("target").merge(
# MAGIC     df_new.alias("source"),
# MAGIC     "target.CustomerID = source.CustomerID"
# MAGIC ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
# MAGIC ```
# MAGIC 
# MAGIC ### 4. Testing and Validation
# MAGIC - Compare row counts with DataStage job
# MAGIC - Validate data transformations match exactly
# MAGIC - Test with production data volumes
# MAGIC - Performance benchmark vs DataStage
# MAGIC 
# MAGIC ### 5. Monitoring and Alerting
# MAGIC - Track rejection rates over time
# MAGIC - Monitor execution duration
# MAGIC - Set alerts for data quality threshold breaches
# MAGIC - Log audit trail for compliance
# MAGIC 
# MAGIC ## Additional Resources
# MAGIC 
# MAGIC - [Delta Lake Documentation](https://docs.delta.io/)
# MAGIC - [Databricks Best Practices](https://docs.databricks.com/optimizations/index.html)
# MAGIC - [PySpark SQL Functions](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/functions.html)
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC **Migration completed successfully! 🎉**
# MAGIC 
# MAGIC This notebook is production-ready and includes:
# MAGIC - ✅ All DataStage transformations translated
# MAGIC - ✅ Data quality constraints enforced
# MAGIC - ✅ Modernized to Delta Lake
# MAGIC - ✅ Performance optimizations applied
# MAGIC - ✅ Comprehensive validation and profiling
# MAGIC - ✅ Detailed execution summary
