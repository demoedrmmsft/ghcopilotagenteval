# Test Artifacts - DataStage Jobs de Ejemplo

Esta carpeta contiene jobs DataStage de ejemplo para validar el agente de migración.

## Artefactos Disponibles

### 1. Simple_Customer_ETL.dsx
**Complejidad**: Básica  
**Propósito**: Validar traducción fundamental de ETL

**Componentes**:
- Sequential File Input (CSV)
- Transformer con derivaciones básicas
- Sequential File Output

**Transformaciones**:
- Concatenación de strings (FirstName + LastName)
- Funciones de string (Trim, Upcase, Downcase)
- Manejo de nulos con defaults
- Cálculos de fecha (YearsFromDate, DaysSince)
- Constraints simples

**Uso**:
```
Pedir al agente: "Migra el job 01_simple_customer_etl.dsx a Databricks"
```

---

### 2. Order_Processing_With_Join.dsx
**Complejidad**: Media  
**Propósito**: Validar joins, lookups y aggregations

**Componentes**:
- Múltiples inputs (Orders, Customers, Products)
- Join Stage (Inner join)
- Lookup Stage (para tabla de referencia)
- Aggregator Stage (GROUP BY con múltiples funciones)
- Sort Stage
- Output

**Patrones**:
- Join de dos tablas grandes
- Lookup de tabla pequeña (reference data)
- Aggregation con SUM, COUNT, AVG, MIN, MAX
- Ordenamiento por múltiples columnas

**Uso**:
```
Pedir al agente: "Migra el job 02_order_processing_join.dsx y optimiza para Spark"
```

---

### 3. Customer_Dimension_SCD2.dsx
**Complejidad**: Alta  
**Propósito**: Validar migración de Slowly Changing Dimensions Type 2

**Componentes**:
- ODBC Source (tabla source)
- ODBC Source (dimensión actual)
- Change Capture Stage
- Transformer con lógica SCD
- Multiple outputs (Insert, Update)
- Stage variables para versionado

**Patrones**:
- Change Data Capture
- SCD Type 2 (mantener historial)
- Surrogate keys
- Effective/End dates
- IsCurrent flags
- Version tracking

**Uso**:
```
Pedir al agente: "Migra el job 03_scd_type2_dimension.dsx a Delta Lake con merge operations"
```

---

### 4. Transaction_Validation_WithErrors.dsx
**Complejidad**: Media-Alta  
**Propósito**: Validar error handling y reject links

**Componentes**:
- Sequential File Input (con modo PERMISSIVE)
- Transformer con Stage Variables complejas
- Múltiples constraints y validaciones
- Reject link para registros inválidos
- Enriquecimiento de buenos registros
- Metadata de errores en rechazos

**Patrones**:
- Data quality validations
- Error categorization (códigos de error)
- Reject links
- Conditional routing
- Stage variables para validaciones complejas
- Error logging y metadata

**Uso**:
```
Pedir al agente: "Migra el job 04_validation_error_handling.dsx con manejo robusto de errores"
```

---

## Casos de Prueba Sugeridos

### Prueba 1: Migración Básica
```
Job: 01_simple_customer_etl.dsx
Validar:
- ✓ Lectura de CSV con opciones correctas
- ✓ Traducción de funciones BASIC a PySpark
- ✓ Manejo de nulos apropiado
- ✓ Escritura con formato correcto
- ✓ Parámetros mapeados a widgets
```

### Prueba 2: Joins y Aggregations
```
Job: 02_order_processing_join.dsx
Validar:
- ✓ Inner join traducido correctamente
- ✓ Lookup convertido a broadcast join
- ✓ Aggregations con todas las funciones (SUM, AVG, etc.)
- ✓ Ordenamiento aplicado
- ✓ Optimizaciones sugeridas
```

### Prueba 3: SCD Type 2
```
Job: 03_scd_type2_dimension.dsx
Validar:
- ✓ Change Capture traducido a Delta merge
- ✓ Lógica de versionado implementada
- ✓ Effective/End dates manejados
- ✓ IsCurrent flag implementado
- ✓ Surrogate keys generadas apropiadamente
```

### Prueba 4: Error Handling
```
Job: 04_validation_error_handling.dsx
Validar:
- ✓ Reject links traducidos a filtros separados
- ✓ Stage variables convertidas a columnas temporales o window functions
- ✓ Validaciones implementadas correctamente
- ✓ Metadata de errores agregada
- ✓ Buenos y malos registros separados apropiadamente
```

---

## Validación de Output

Para cada migración, verificar:

1. **Estructura del Notebook**
   - Header con metadata
   - Sección de parámetros (widgets)
   - Imports y configuración
   - Stages documentados con markdown
   - Validaciones finales

2. **Código PySpark**
   - Usa funciones nativas (no UDFs innecesarias)
   - Optimizaciones aplicadas (broadcast, partitioning)
   - Manejo de errores implementado
   - Logging agregado

3. **Documentación**
   - Mapeo de DataStage stages a código PySpark
   - Decisiones de diseño explicadas
   - Notas sobre diferencias con DataStage
   - Próximos pasos sugeridos

4. **Best Practices**
   - Delta Lake usado apropiadamente
   - Configuración de Spark optimizada
   - Secrets para credentials
   - Métricas capturadas

---

## Extensión de Test Artifacts

Para agregar más casos de prueba, considerar:

- **Pivot/Unpivot operations**
- **Change Capture con Delete tracking**
- **Complex Window Functions** (Running totals, Lag/Lead)
- **Funnel (UNION) stages**
- **Remove Duplicates con Keep First/Last**
- **Row Generator con patrones complejos**
- **Surrogate Key Generation**
- **Checksum calculations**
- **Complex BASIC Routines** (requieren reescritura completa)
- **Shared Containers** (funciones reutilizables)
- **Parallel jobs** (vs Sequential jobs)
- **Database stages** con UPSERT operations
- **XML/JSON parsing**
- **Hierarchical stages** (Complex Flat File)

---

## Notas Importantes

### Limitaciones de Simulación
Estos DSX son representaciones simplificadas del formato DataStage. En producción real:
- DSX son binarios más complejos
- Incluyen más metadata de compilación
- Tienen optimizaciones de runtime
- Contienen información de deployment

### Para Parsear DSX Reales
Si trabajas con DSX reales de producción:
1. Usar IBM InfoSphere Metadata Workbench
2. Exportar a formato legible (XML)
3. O usar DataStage Designer para inspección visual
4. Considerar usar IBM DataStage Flow Designer (web-based)

### Validación con Datos Reales
Al migrar jobs reales:
1. Ejecutar job DataStage original con dataset de prueba
2. Capturar output y checksums
3. Ejecutar notebook Databricks migrado con mismo input
4. Comparar outputs (counts, checksums, sample records)
5. Validar performance (tiempo de ejecución)
