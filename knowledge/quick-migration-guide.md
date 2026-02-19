# Guía Rápida de Migración DataStage → Databricks

Esta guía proporciona un proceso paso a paso para migrar jobs de DataStage a Databricks usando el agente de Copilot.

## 🎯 Proceso de Migración (30 minutos - 2 horas por job)

### Fase 1: Preparación (5-10 min)

#### 1.1 Exportar Job DataStage
```bash
# Opción A: Desde DataStage Designer
File → Export → DataStage Components → Select Job → Export as DSX

# Opción B: Comando (DataStage Server)
dsadmin -DSExport -domain <domain> -user <user> -password <pass> \
  -project <project> -jobname <jobname> -file <output.dsx>
```

#### 1.2 Reunir Información del Job
Documentar:
- [ ] Propósito del job
- [ ] Frecuencia de ejecución (diaria, horaria, etc.)
- [ ] Volumen de datos típico
- [ ] Dependencias (jobs upstream/downstream)
- [ ] Parámetros y sus valores típicos
- [ ] SLAs y requisitos de performance

#### 1.3 Preparar Entorno Databricks
- [ ] Crear o seleccionar workspace Databricks
- [ ] Configurar cluster (tamaño apropiado)
- [ ] Configurar Delta Lake paths (si aplica)
- [ ] Configurar Secrets para credentials
- [ ] Crear schemas en Unity Catalog (recomendado)

---

### Fase 2: Migración con Agente (10-30 min)

#### 2.1 Usar GitHub Copilot Chat

Copiar el contenido del archivo DSX y pegar en el chat:

```
@workspace Tengo este job DataStage que necesito migrar a Databricks:

[pegar contenido del DSX aquí]

Por favor:
1. Analiza todos los stages y su flujo
2. Genera un notebook Databricks completo
3. Incluye optimizaciones para Spark
4. Agrega logging y error handling
5. Documenta las decisiones de diseño
```

#### 2.2 Alternativa: Proporcionar Descripción

Si no tienes el DSX, describe el job:

```
@workspace Necesito migrar un job DataStage con esta estructura:

- Input: Lee CSV con órdenes desde S3
- Join: Une con tabla de clientes (inner join)
- Lookup: Busca productos en tabla de referencia pequeña
- Aggregator: Suma ventas por cliente y mes
- Output: Escribe a Delta Lake

Incluye:
- Parámetros para rutas de input/output
- Validaciones de calidad de datos
- Manejo de errores
- Monitoreo
```

#### 2.3 Revisión del Código Generado

El agente te dará:
1. **Análisis del Job**: Resumen de stages y complejidad
2. **Notebook Databricks**: Código PySpark completo
3. **Notas de Migración**: Decisiones importantes
4. **Recomendaciones**: Optimizaciones y tests

Revisar:
- [ ] Todas las transformaciones están migradas
- [ ] Joins tienen estrategia correcta (broadcast vs shuffle)
- [ ] Particionamiento es apropiado
- [ ] Error handling está implementado
- [ ] Parámetros son configurable (widgets)

---

### Fase 3: Implementación (15-30 min)

#### 3.1 Crear Notebook en Databricks

1. Ir a Databricks Workspace
2. Create → Notebook
3. Nombrar: `[JobName]_Migrated`
4. Copiar código generado por el agente
5. Guardar

#### 3.2 Configurar Parámetros

```python
# Crear widgets para todos los parámetros DataStage
dbutils.widgets.text("input_path", "", "Input Path")
dbutils.widgets.text("output_path", "", "Output Path")
dbutils.widgets.dropdown("load_type", "FULL", ["FULL", "INCREMENTAL"])

# Si usas secrets
input_path = dbutils.secrets.get("scope", "input_path_key")
# O desde widgets
input_path = dbutils.widgets.get("input_path")
```

#### 3.3 Ajustar Configuración de Spark

```python
# Adaptar según tamaño de datos
spark.conf.set("spark.sql.shuffle.partitions", "200")  # Ajustar según volumen
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
```

#### 3.4 Crear Tablas Delta (si aplica)

```python
# Si usas Unity Catalog
spark.sql("""
  CREATE TABLE IF NOT EXISTS catalog.schema.table_name (
    column1 STRING,
    column2 INT,
    ...
  )
  USING DELTA
  PARTITIONED BY (partition_column)
""")
```

---

### Fase 4: Testing (15-30 min)

#### 4.1 Test con Dataset Pequeño

```python
# Usar subset de datos para testing rápido
test_input = "/path/to/test_data_small.csv"
test_output = "/tmp/test_output"

# Ejecutar notebook con test data
```

#### 4.2 Validar Output

```python
# Comparar con output DataStage si disponible
df_databricks = spark.read.format("delta").load(test_output)
df_datastage = spark.read.csv("/path/to/datastage_output")

# Comparar counts
print(f"Databricks count: {df_databricks.count()}")
print(f"DataStage count: {df_datastage.count()}")

# Comparar sample
display(df_databricks.limit(10))
display(df_datastage.limit(10))

# Comparar checksums (columnas numéricas)
databricks_sum = df_databricks.agg({"numeric_col": "sum"}).collect()[0][0]
datastage_sum = df_datastage.agg({"numeric_col": "sum"}).collect()[0][0]
print(f"Sum difference: {databricks_sum - datastage_sum}")
```

#### 4.3 Tests de Calidad de Datos

```python
# Verificar nulls en columnas clave
null_checks = df_databricks.select(
    *[F.sum(F.col(c).isNull().cast("int")).alias(f"{c}_nulls") 
      for c in df_databricks.columns]
).collect()[0].asDict()

for col, null_count in null_checks.items():
    if null_count > 0:
        print(f"⚠️ {col}: {null_count} null values")

# Verificar duplicados
dup_count = df_databricks.groupBy("key_column").count() \
    .filter(F.col("count") > 1).count()
if dup_count > 0:
    print(f"⚠️ Found {dup_count} duplicate keys")
```

---

### Fase 5: Optimización (Opcional, 10-20 min)

#### 5.1 Analizar Performance

```python
# Ejecutar con datos completos
# Revisar Spark UI para identificar bottlenecks

# Queries lentas
spark.sql("EXPLAIN COST SELECT * FROM ...").show()

# Identificar shuffles costosos
# Ver Spark UI → Stages → identificar stages con mucho shuffle
```

#### 5.2 Aplicar Optimizaciones

```python
# Si hay join lento y una tabla es pequeña
df_result = df_large.join(broadcast(df_small), on="key")

# Si hay aggregation lenta
# Reparticionar antes del groupBy
df.repartition("partition_key").groupBy("partition_key").agg(...)

# Compactar archivos Delta
spark.sql("OPTIMIZE delta.`/path/to/table`")
spark.sql("OPTIMIZE delta.`/path/to/table` ZORDER BY (commonly_filtered_column)")
```

#### 5.3 Tuning de Cluster

- Aumentar workers si proceso es muy lento
- Usar instance types con más memoria si hay OOM errors
- Habilitar autoscaling para cargas variables

---

### Fase 6: Deployment (10-15 min)

#### 6.1 Crear Job en Databricks

1. Databricks UI → Workflows → Create Job
2. Configurar:
   - **Task name**: Descriptivo
   - **Type**: Notebook
   - **Notebook path**: Seleccionar notebook migrado
   - **Cluster**: Seleccionar o crear
   - **Parameters**: Agregar key-value pairs para widgets

3. Schedule (si aplica):
   - **Trigger type**: Scheduled
   - **Cron expression**: Equivalente al schedule DataStage

#### 6.2 Configurar Alertas

```python
# En el notebook, agregar al final
success_message = f"""
✅ Job completado exitosamente
- Records processed: {record_count}
- Execution time: {execution_time} seconds
- Output path: {output_path}
"""

# Enviar notificación (ejemplo con Slack webhook)
import requests
requests.post(slack_webhook_url, json={"text": success_message})

# O usar Databricks Job notifications
# (configurar en Job UI → Alerts)
```

#### 6.3 Monitoreo

- Configurar alertas en Job failures
- Revisar métricas regularmente
- Guardar logs en tabla para análisis:

```python
# Al final del notebook
job_metrics = spark.createDataFrame([{
    "job_name": "customer_etl",
    "execution_date": datetime.now(),
    "status": "SUCCESS",
    "records_processed": record_count,
    "execution_time_sec": execution_time,
    "errors_count": error_count
}])

job_metrics.write.format("delta").mode("append") \
    .saveAsTable("catalog.monitoring.job_metrics")
```

---

## 🎓 Tips y Mejores Prácticas

### Durante la Migración

1. **Empezar Simple**: Migrar jobs pequeños primero para familiarizarte
2. **Documentar Todo**: Decisiones, diferencias con DataStage, problemas encontrados
3. **Comparar Outputs**: Validar con data de prueba contra output DataStage original
4. **Optimizar Gradualmente**: Funcionalidad primero, optimización después
5. **Reusar Patrones**: Crear templates para patrones comunes (SCD, validation, etc.)

### Patrones Comunes a Conocer

| DataStage Pattern | Databricks Pattern |
|------------------|-------------------|
| Sequential File → Dataset | CSV/Parquet → Delta Lake |
| Lookup (small table) | Broadcast Join |
| Join (large tables) | Repartition Join |
| Aggregator | groupBy().agg() |
| Sort | orderBy() |
| Remove Duplicates | dropDuplicates() |
| Change Capture + SCD2 | Delta Lake Merge |
| Reject Links | filter() + separate DataFrames |
| Stage Variables | Columnas temporales o Window functions |

### Errores Comunes a Evitar

❌ **No optimizar joins**
```python
# Lento si ambas tablas son grandes
df1.join(df2, on="key")  
```
✅ **Usar broadcast para tablas pequeñas**
```python
df_large.join(broadcast(df_small), on="key")
```

❌ **No particionar antes de writes**
```python
df.write.save(path)  # Miles de archivos pequeños
```
✅ **Coalesce para consolidar**
```python
df.coalesce(50).write.save(path)  # 50 archivos
```

❌ **Usar Python UDFs innecesariamente**
```python
@udf
def calculate(x):
    return x * 2
df.withColumn("result", calculate("value"))
```
✅ **Usar funciones nativas de Spark**
```python
df.withColumn("result", F.col("value") * 2)
```

---

## 📊 Checklist de Migración Completa

### Funcionalidad
- [ ] Todas las transformaciones migradas
- [ ] Joins/Lookups correctamente implementados
- [ ] Aggregations con todas las funciones requeridas
- [ ] Filtros y constraints aplicados
- [ ] Error handling equivalente a DataStage

### Performance
- [ ] Particionamiento optimizado
- [ ] Broadcast joins donde apropiado
- [ ] Shuffle minimizado
- [ ] AQE (Adaptive Query Execution) habilitado
- [ ] Delta Lake usado para tablas intermedias

### Operaciones
- [ ] Parámetros como widgets configurables
- [ ] Secrets para credentials
- [ ] Logging implementado
- [ ] Métricas capturadas
- [ ] Alertas configuradas

### Testing
- [ ] Comparado con output DataStage
- [ ] Validaciones de calidad ejecutadas
- [ ] Tests con volúmenes representativos
- [ ] Performance aceptable
- [ ] Documentación actualizada

### Deployment
- [ ] Job creado en Databricks
- [ ] Schedule configurado
- [ ] Dependencias documentadas
- [ ] Runbook creado
- [ ] Team entrenado

---

## 🆘 Troubleshooting

### Problema: Job falla con Out of Memory

**Solución**:
```python
# 1. Reducir shuffle partitions si dataset es pequeño
spark.conf.set("spark.sql.shuffle.partitions", "50")

# 2. Aumentar memoria del driver/executors
# (configurar en cluster settings)

# 3. Procesar en batches
for batch in batches:
    process(batch)
    # Limpiar cache entre batches
    spark.catalog.clearCache()
```

### Problema: Job muy lento comparado con DataStage

**Investigar**:
1. Revisar Spark UI para identificar stage lento
2. Verificar si hay data skew (una partición mucho mayor que otras)
3. Verificar si broadcast join falló (tabla muy grande)
4. Revisar número de particiones

**Soluciones**:
```python
# Repartition por key que distribuya uniformemente
df.repartition(200, "key_column")

# Forzar broadcast si tabla es <10GB
df.join(broadcast(df_small), on="key")

# Habilitar AQE para optimizaciones automáticas
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
```

### Problema: Resultados diferentes a DataStage

**Verificar**:
1. Orden de operaciones (sort puede cambiar resultados)
2. Manejo de nulls (comportamiento diferente)
3. Conversiones de tipo (casting)
4. Truncamiento de fechas/timestamps
5. Funciones de string (índices base-0 vs base-1)

**Debug**:
```python
# Comparar stage por stage
df_after_stage1 = ...
display(df_after_stage1.limit(10))

# Comparar checksums
df.select(F.sum(F.hash(*df.columns))).show()
```

---

## 📚 Recursos Adicionales

- **Knowledge Base**: `knowledge/` folder con patrones detallados
- **Test Artifacts**: `test-artifacts/` con jobs de ejemplo
- **Databricks Docs**: [docs.databricks.com](https://docs.databricks.com)
- **Delta Lake Docs**: [delta.io](https://delta.io)
- **PySpark API**: [spark.apache.org/docs/latest/api/python](https://spark.apache.org/docs/latest/api/python)

---

## 💡 Próximos Pasos Después de la Migración

1. **Monitoreo Continuo** (Semanas 1-4)
   - Comparar outputs con DataStage daily
   - Monitorear performance y costos
   - Ajustar configuraciones según necesidad

2. **Optimización** (Mes 1-2)
   - Identificar jobs que consumen más recursos
   - Aplicar optimizaciones avanzadas
   - Implementar caching strategies

3. **Decommissioning DataStage** (Mes 2-6)
   - Migrar jobs restantes
   - Ejecutar ambos sistemas en paralelo
   - Validar completamente antes de shutdown
   - Documentar lessons learned
