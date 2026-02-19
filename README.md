# 🤖 GitHub Copilot Agent: DataStage → Databricks Migration

Un agente declarativo de GitHub Copilot especializado en migrar jobs de IBM DataStage a Azure Databricks, generando código PySpark optimizado, notebooks Databricks completos y siguiendo mejores prácticas de ingeniería de datos.

> **Tipo de Agente**: GitHub Copilot Agent (Declarativo con `agent.yml`)  
> **Sin servidor requerido** - Funciona directamente en VS Code

[![GitHub Copilot](https://img.shields.io/badge/GitHub%20Copilot-Agent-blue?logo=github)](https://github.com/features/copilot)
[![Databricks](https://img.shields.io/badge/Databricks-Ready-orange?logo=databricks)](https://databricks.com)
[![PySpark](https://img.shields.io/badge/PySpark-3.x-yellow?logo=apache-spark)](https://spark.apache.org/pyspark.html)

---

## 📑 Tabla de Contenido

- [Características](#características)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Instalación Rápida](#instalación)
- [Cómo Usar el Agente](#cómo-usar-el-agente)
- [Ejemplos de Uso](#ejemplos-de-uso)
- [Capacidades del Agente](#capacidades-del-agente)
- [Guías Detalladas](#guías-detalladas)
- [Troubleshooting](#troubleshooting)
- [Contribuir](#contribuir)
- [Testing y Validación](#testing-y-validación)
- [Mejores Prácticas](#mejores-prácticas)
- [Troubleshooting](#troubleshooting)
- [Contribuir](#contribuir)

---

## 🎯 Características

### Análisis Inteligente de Jobs DataStage
- ✅ Parsea archivos DSX (XML de DataStage)
- ✅ Identifica tipos de stages (Sequential File, Transformer, Join, Aggregator, etc.)
- ✅ Mapea flujo de datos entre stages
- ✅ Extrae transformaciones, filtros y lógica de negocio

### Generación de Código PySpark Optimizado
- ✅ Traduce stages DataStage a operaciones PySpark equivalentes
- ✅ Implementa transformaciones complejas (derivations, constraints)
- ✅ Convierte lógica BASIC de DataStage a Python/PySpark
- ✅ Optimiza para procesamiento distribuido
- ✅ Implementa error handling robusto

### Notebooks Databricks Estructurados
- ✅ Notebooks bien documentados con markdown
- ✅ Widgets para parámetros configurables
- ✅ Logging y monitoreo integrado
- ✅ Validaciones de calidad de datos
- ✅ Tests unitarios sugeridos

### Mejores Prácticas
- ✅ Aprovecha Delta Lake para ACID transactions
- ✅ Implementa particionamiento eficiente
- ✅ Usa broadcast joins inteligentemente
- ✅ Aplica optimizaciones de Spark (AQE, caching, etc.)
- ✅ Maneja errores y rechazos apropiadamente

---

## 📁 Estructura del Proyecto

```
agentcopiloteval/
├── agent.yml                            # ⭐ Configuración del agente (declarativa)
├── README.md                            # Este archivo
├── knowledge/                           # Base de conocimiento del agente
│   ├── datastage-components.md         # Catálogo completo de stages
│   ├── migration-patterns.md           # Patrones con código completo
│   ├── databricks-best-practices.md    # Optimizaciones Databricks
│   └── quick-migration-guide.md        # Guía paso a paso
├── test-artifacts/                      # Jobs DataStage de prueba (.dsx)
│   ├── 01_simple_customer_etl.dsx      # ETL básico
│   ├── 02_order_processing_join.dsx    # Joins y agregaciones
│   ├── 03_scd_type2_dimension.dsx      # SCD Type 2
│   ├── 04_validation_error_handling.dsx # Manejo de errores
│   └── README.md                        # Documentación de casos
└── examples/                            # Notebooks Databricks de ejemplo
    └── sample_migrated_notebook.py      # Ejemplo completo migrado
```

### Archivos Clave

- **`agent.yml`**: Configuración declarativa del agente con instrucciones, capacidades, y referencias a knowledge base
- **`knowledge/*.md`**: Base de conocimiento con patrones, componentes, y mejores prácticas
- **`test-artifacts/*.dsx`**: Jobs de ejemplo para validar el agente
- **`examples/`**: Notebooks Databricks ya migrados como referencia

---

## 🚀 Cómo Usar el Agente

### Requisitos Previos

1. **GitHub Copilot** instalado y activado en VS Code
2. **VS Code** versión reciente
3. Este **workspace abierto** en VS Code

### Setup Rápido

```bash
# 1. Clona el repositorio
git clone https://github.com/your-org/datastage-migration-agent.git
cd datastage-migration-agent

# 2. Abre en VS Code
code .

# 3. ¡Listo! El agente se carga automáticamente desde agent.yml
```

### Método 1: Migración Directa (Archivo .dsx)

1. **Abre GitHub Copilot Chat**: `Ctrl+Shift+I` (Windows/Linux) o `Cmd+Shift+I` (Mac)

2. **Invoca el agente** con `@workspace`:

```
@workspace Migra el job test-artifacts/01_simple_customer_etl.dsx a Databricks
```

3. **El agente genera**:
   - ✅ Análisis del job (stages, flujo de datos, parámetros)
   - 📓 Notebook Databricks completo con código PySpark
   - 💡 Recomendaciones de optimización específicas
   - 📝 Próximos pasos de implementación

### Método 2: Análisis sin Migración

Para solo analizar complejidad y obtener estimaciones:

### Método 2: Análisis sin Migración

Para solo analizar complejidad y obtener estimaciones:

```
@workspace Analiza la complejidad del job test-artifacts/03_scd_type2_dimension.dsx
```

**El agente responde con**:
- Nivel de complejidad: Low, Medium, High, Very High
- Estimación de esfuerzo en horas
- Lista de stages y tipos
- Desafíos potenciales identificados
- Recomendaciones de abordaje

### Método 3: Consultas sobre Componentes

Para entender componentes DataStage y sus equivalentes:

```
@workspace Explica cómo migrar un stage Aggregator a PySpark
```

```
@workspace ¿Cómo implemento SCD Type 2 en Databricks?
```

```
@workspace Dame un ejemplo de traducir expresiones BASIC a PySpark
```
3. Calcular LineTotal = Quantity * UnitPrice
4. Agregar por Cliente y Mes:
   - Total Orders
   - Revenue Total
   - Avg Order Value
5. Filtrar solo órdenes completadas
6. Ordenar por Revenue descendente

**Output:**
- Delta Lake table particionada por año/mes

Incluye validaciones de calidad de datos y error handling.
```

### Método 3: Migrar desde Test Artifacts

Usa los jobs de ejemplo incluidos:

```
@workspace Migra el job test-artifacts/02_order_processing_join.dsx 
a Databricks con todas las optimizaciones recomendadas
```

---

## 💡 Ejemplos de Uso

### Ejemplo 1: ETL Simple

**Prompt**:
```
@workspace Migra el job test-artifacts/01_simple_customer_etl.dsx
```

**El agente generará**:
- Notebook con:
  - Lectura de CSV con spark.read
  - Transformaciones con withColumn()
  - Limpieza de datos (trim, upper, null handling)
  - Escritura a Delta Lake
  - Validaciones

### Ejemplo 2: Joins y Aggregations

**Prompt**:
```
@workspace Analiza y migra test-artifacts/02_order_processing_join.dsx
Optimiza especialmente los joins y explain las decisiones.
```

**El agente generará**:
- Análisis de cada join (inner, lookup)
- Código con broadcast join para tabla pequeña
- Aggregations con groupBy().agg()
- Explicación de por qué eligió broadcast vs shuffle join

### Ejemplo 3: SCD Type 2

**Prompt**:
```
@workspace Migra test-artifacts/03_scd_type2_dimension.dsx
Usa Delta Lake merge operations para implementar SCD Type 2
```

**El agente generará**:
- Implementación con DeltaTable.merge()
- Lógica para cerrar registros antiguos
- Inserción de nuevas versiones
- Manejo de Effective/End dates e IsCurrent flags

### Ejemplo 4: Error Handling

**Prompt**:
```
@workspace Migra test-artifacts/04_validation_error_handling.dsx
Asegúrate de manejar todos los errores y generar logs detallados
```

**El agente generará**:
- DataFrames separados para buenos y malos registros
- Validaciones implementadas con filter()
- Metadata de errores agregada
- Logging comprehensivo

### Ejemplo 5: Optimización de Job Existente

**Prompt**:
```
@workspace Tengo este notebook Databricks [pegar código]
que fue migrado desde DataStage pero es muy lento.
Analízalo y sugiere optimizaciones.
```

**El agente analizará**:
- Joins y sugerirá broadcast donde apropiado
- Particionamiento y sugerirá repartition
- Shuffles innecesarios
- Oportunidades de caching
- Configuraciones de Spark

---

## 🎓 Capacidades del Agente

### 1. Traducción de Stages DataStage

| DataStage Stage | PySpark Equivalent | Agente Implementa |
|----------------|-------------------|-------------------|
| Sequential File | `spark.read.csv()` | ✅ Con todas las opciones |
| Transformer | `withColumn()`, `filter()` | ✅ Con expresiones complejas |
| Join | `join()` | ✅ Con estrategia optimizada |
| Lookup | `broadcast(df).join()` | ✅ Auto-detecta tablas pequeñas |
| Aggregator | `groupBy().agg()` | ✅ Con todas las funciones |
| Sort | `orderBy()` | ✅ Con múltiples keys |
| Remove Duplicates | `dropDuplicates()` | ✅ Con subset de columnas |
| Change Capture | Delta CDC | ✅ Con merge operations |
| Funnel | `union()` | ✅ Con unionByName |

### 2. Traducción de Expresiones BASIC

El agente traduce automáticamente:

```basic
# DataStage BASIC
If IsNull(Column1) Then "DEFAULT" Else Column1
Trim(Upcase(FirstName)) : " " : Trim(Upcase(LastName))
Column1[1,10]
YearsFromDate(BirthDate)
```

```python
# PySpark generado por el agente
F.when(F.col("Column1").isNull(), F.lit("DEFAULT")).otherwise(F.col("Column1"))
F.concat(F.trim(F.upper("FirstName")), F.lit(" "), F.trim(F.upper("LastName")))
F.substring("Column1", 1, 10)
F.floor(F.months_between(F.current_date(), F.col("BirthDate")) / 12)
```

### 3. Optimizaciones Automáticas

El agente aplica:

- ✅ **Broadcast Joins**: Para tablas pequeñas (<10GB)
- ✅ **Particionamiento**: Basado en keys y volumen de datos
- ✅ **Delta Lake**: Para todas las tablas intermedias
- ✅ **AQE (Adaptive Query Execution)**: Habilitado por default
- ✅ **Z-Ordering**: Para queries con filtros comunes
- ✅ **Caching**: Para DataFrames reutilizados

### 4. Patrones Avanzados

El agente maneja:

- ✅ **SCD Type 2**: Con Delta merge y lógica de versionado
- ✅ **Change Data Capture**: Detección de cambios
- ✅ **Error Handling**: Reject links → DataFrames separados
- ✅ **Stage Variables**: Convertidos a window functions o columnas temporales
- ✅ **Complex Constraints**: Múltiples validaciones con metadata de errores

---

## 🧪 Testing y Validación

### Usar Test Artifacts

El proyecto incluye 4 jobs DataStage de ejemplo:

```bash
test-artifacts/
├── 01_simple_customer_etl.dsx           # Básico
├── 02_order_processing_join.dsx         # Joins
├── 03_scd_type2_dimension.dsx           # SCD
└── 04_validation_error_handling.dsx     # Errors
```

### Proceso de Validación

1. **Migrar con el agente**:
```
@workspace Migra test-artifacts/01_simple_customer_etl.dsx
```

2. **Copiar código a Databricks notebook**

3. **Ejecutar con datos de prueba**

4. **Validar output**:
```python
# Comparar counts
print(f"Records: {df_output.count()}")

# Verificar schema
df_output.printSchema()

# Revisar sample
display(df_output.limit(10))

# Validaciones de calidad
null_counts = df_output.select(
    *[F.sum(F.col(c).isNull().cast("int")).alias(c) 
      for c in df_output.columns]
).collect()[0].asDict()

for col, nulls in null_counts.items():
    if nulls > 0:
        print(f"⚠️ {col}: {nulls} nulls")
```

### Checklist de Validación

Para cada migración, verificar:

- [ ] 🎯 **Funcionalidad**: Todas las transformaciones migradas
- [ ] 📊 **Output Correcto**: Counts y samples coinciden con expectativa
- [ ] ⚡ **Performance**: Tiempo de ejecución aceptable
- [ ] 🛡️ **Error Handling**: Rechazos manejados apropiadamente
- [ ] 📝 **Documentación**: Código bien comentado y explicado
- [ ] 🔧 **Optimizaciones**: Broadcast, partitioning aplicados
- [ ] ✅ **Validaciones**: Data quality checks implementados

---

## 📚 Mejores Prácticas

### 1. Preparación

**Antes de migrar, reunir**:
- Archivo DSX exportado desde DataStage
- Descripción del job y su propósito
- Volumen típico de datos
- Requisitos de performance (SLAs)
- Jobs upstream/downstream (dependencias)

### 2. Iteración

**Proceso recomendado**:
1. Migrar funcionalidad primero (hacer que funcione)
2. Validar con datos de prueba
3. Optimizar performance
4. Agregar monitoring y alerts
5. Documentar

### 3. Comparación con DataStage

**Para validar migración**:
```python
# Ejecutar DataStage job con test data
# Capturar output

# Ejecutar notebook Databricks con mismo input
# Comparar outputs:

# - Counts
# - Checksums de columnas numéricas
# - Sample de registros
# - Schema (tipos de datos)
```

### 4. Monitoreo Post-Migración

**Implementar**:
```python
# Métricas de ejecución
metrics = {
    "job_name": "migrated_job",
    "start_time": start_time,
    "end_time": end_time,
    "duration_sec": duration,
    "records_input": input_count,
    "records_output": output_count,
    "records_rejected": reject_count,
    "status": "SUCCESS"
}

# Guardar en tabla de métricas
spark.createDataFrame([metrics]).write \
    .mode("append").saveAsTable("monitoring.job_metrics")
```

---

## 🐛 Troubleshooting

### Problema: El agente no encuentra mis archivos DSX

**Solución**: 
- Asegúrate de que el archivo DSX esté en el workspace
- O pega el contenido completo en el chat
- O describe el job en detalle

### Problema: El código generado no funciona

**Revisar**:
1. **Paths**: Ajustar rutas de input/output para tu entorno
2. **Credentials**: Configurar secrets de Databricks
3. **Schema**: Verificar que columnas existan con nombres correctos
4. **Tipos de Datos**: Ajustar casting si necesario

**Pedir ayuda al agente**:
```
@workspace Tengo este error al ejecutar el notebook:
[pegar error]

El código es:
[pegar código relevante]

¿Cómo lo soluciono?
```

### Problema: Performance muy lento

**Pedir optimizaciones**:
```
@workspace Este notebook está muy lento:
[pegar código]

Datos:
- Input: 100GB daily
- Cluster: 10 workers, 16GB RAM cada uno
- Tarda: 2 horas (queremos < 30 min)

¿Qué optimizaciones recomiendas?
```

El agente sugerirá:
- Broadcast joins
- Reparticionamiento
- Caching estratégico
- Configuraciones de Spark
- Ajustes de cluster

### Problema: Resultados diferentes a DataStage

**Investigar con el agente**:
```
@workspace Los resultados no coinciden con DataStage:

DataStage output:
- Count: 10,000
- Sum(amount): 1,000,000

Databricks output:
- Count: 9,500
- Sum(amount): 950,000

¿Qué puede estar causando la diferencia?
```

El agente analizará:
- Manejo de nulls
- Filtros/constraints
- Joins (inner vs left)
- Orden de operaciones
- Casting de tipos

---

## 🤝 Contribuir

### Agregar Nuevos Patrones

Para extender el agente con nuevos patrones:

1. Agregar documentación en `knowledge/migration-patterns.md`
2. Crear ejemplo en `test-artifacts/`
3. Actualizar `.github-copilot-instructions.md` si necesario

### Reportar Issues

Si encuentras problemas:
1. Describir el job DataStage (o incluir DSX)
2. Compartir el prompt usado
3. Incluir el código generado
4. Describir el problema encontrado
5. Incluir logs de error si aplica

### Sugerir Mejoras

Ideas para mejorar el agente:
- Nuevos tipos de stages DataStage
- Patrones de optimización adicionales
- Mejor manejo de casos edge
- Documentación mejorada

---

## 📖 Documentación Adicional

### Guías Detalladas
- [Guía Rápida de Migración](knowledge/quick-migration-guide.md) - Proceso paso a paso
- [Catálogo de Componentes DataStage](knowledge/datastage-components.md) - Todos los stages
- [Patrones de Migración](knowledge/migration-patterns.md) - Ejemplos detallados
- [Mejores Prácticas Databricks](knowledge/databricks-best-practices.md) - Optimizaciones

### Test Artifacts
- [README de Test Artifacts](test-artifacts/README.md) - Documentación de jobs de ejemplo

### Recursos Externos
- [Databricks Documentation](https://docs.databricks.com)
- [Delta Lake Documentation](https://docs.delta.io)
- [PySpark API Reference](https://spark.apache.org/docs/latest/api/python/)
- [IBM DataStage Documentation](https://www.ibm.com/docs/en/iis)

---

## 🎯 Próximos Pasos

### Para Empezar

1. ✅ **Familiarízate**: Lee el [Quick Migration Guide](knowledge/quick-migration-guide.md)
2. ✅ **Prueba**: Migra uno de los test artifacts
3. ✅ **Compara**: Valida el output generado
4. ✅ **Migra**: Empieza con un job DataStage real simple
5. ✅ **Itera**: Mejora y optimiza

### Roadmap

Próximas mejoras planeadas:
- [ ] Soporte para DataStage Parallel Jobs
- [ ] Integración con LineageGraph para dependency mapping
- [ ] Templates de notebooks reutilizables
- [ ] Auto-generación de tests unitarios
- [ ] Comparador automático DataStage vs Databricks output
- [ ] Estimador de costos Databricks

---

## 📄 Licencia

Este proyecto es un agente educational/evaluativo para GitHub Copilot.

---

## ✨ Agradecimientos

Este agente fue creado para facilitar la migración de DataStage a Databricks, aprovechando las capacidades avanzadas de GitHub Copilot para generar código de alta calidad, optimizado y bien documentado.

**¡Feliz Migración! 🚀**

---

## 📞 Soporte

Para preguntas o ayuda:
1. Consulta la [documentación](knowledge/)
2. Revisa los [test artifacts](test-artifacts/)
3. Pregunta al agente directamente con `@workspace`

**Ejemplo de pregunta al agente**:
```
@workspace Tengo dudas sobre cómo migrar un Aggregator stage 
con múltiples grouping keys y stage variables. ¿Puedes explicarme 
el patrón recomendado con ejemplos?
```
