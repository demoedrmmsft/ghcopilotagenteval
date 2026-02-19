# 📋 Recomendaciones de Mejora - Agentes GitHub Copilot
## Proyecto: Migración DataStage → Databricks

**Fecha de Análisis:** 19 de Febrero, 2026  
**Versión:** 1.0  
**Estado:** Para Revisión e Implementación

---

## 📊 RESUMEN EJECUTIVO

| Categoría | Prioridad | Impacto | Archivos Afectados |
|-----------|-----------|---------|-------------------|
| Longitud Excesiva de Prompts | 🔴 Alta | Alto | job-ds-migration.agent.md |
| Repetición y Duplicación | 🟡 Media | Medio | job-ds-migration.agent.md, workflow-ds-migration.agent.md |
| Prompts Simples Subdesarrollados | 🟡 Media | Medio | migrar-jobs.prompt.md, create-cleaner.prompt.md |
| Manejo de Errores Insuficiente | 🟡 Media | Medio | job-ds-migration.agent.md |
| Inconsistencias en Metadata | 🟢 Baja | Bajo | workflow-ds-migration.agent.md, nb-cleaner-creator.agent.md |

**Total de Problemas Identificados:** 5 categorías principales  
**Total de Archivos a Modificar:** 5 archivos

---

## 🔴 PRIORIDAD ALTA

### 1. LONGITUD EXCESIVA DE PROMPTS EN DELEGACIÓN

#### 📍 Ubicación
**Archivo:** `.github/agents/job-ds-migration.agent.md`  
**Líneas:** 85-237 (~152 líneas)  
**Sección:** Template de delegación con `#runSubagent`

#### ❌ Problema Actual
El template de delegación a subagentes es excesivamente largo:
- **152 líneas** de instrucciones en cada delegación
- Incluye reglas completas que ya existen en archivos separados
- Riesgo de "dilución de instrucciones" donde el LLM pierde foco en lo crítico
- Mayor costo de tokens por cada job procesado
- Dificulta mantenimiento (cambios deben hacerse en múltiples lugares)

**Fragmento problemático (líneas 85-237):**
```markdown
#runSubagent "Instrucciones: Actúa como el Especialista Técnico de Migración. 
Tu objetivo es migrar exclusivamente el Job [NOMBRE_JOB_XML]. 

PASOS OBLIGATORIOS:

1. Lee las reglas técnicas COMPLETAS en .github/instructions/parallel_rules/:
   - 02_analysis_rules.md (Análisis de XML)
   - 03_env_rules.md (Parámetros constantes)
   - 04_notebook_rules.md (Creación de notebooks)
     → ⚠️ PRIMERO leer: "ANTI-PATTERNS" y "API REFERENCE"
   - 05_sql_rules.md (Creación de archivos SQL)
   - 06_migration_process_rules.md (Proceso completo)
   - 07_dataset_rules.md (Manejo de datasets)

2. Analiza el XML usando Extracción Semántica Selectiva:
   - Fase 1: Extraer IdentList y crear mapa de stages
   - Fase 2: Extracción selectiva por tipo de stage
   - Fase 3: Validación semántica (100% de stages documentados)

3. Genera TODOS los artefactos obligatorios:
   [... 100+ líneas más de instrucciones detalladas ...]
```

#### ✅ Solución Propuesta

**Reemplazar líneas 85-237 con versión condensada:**

```markdown
#runSubagent "MIGRACIÓN DE JOB: [NOMBRE_JOB_XML]

**CONTEXTO OBLIGATORIO:**
Lee PRIMERO antes de cualquier acción:
- `.github/instructions/parallel_rules/06_migration_process_rules.md` (Proceso completo - PRINCIPAL)
- Este archivo ya contiene referencias a las reglas 02, 03, 04, 05, 07

**OBJETIVO:**
Generar TODOS los artefactos para el job [NOMBRE_JOB]:
1. `migrations/XML_ANALISIS/[NOMBRE_JOB]_ANALYSIS.md`
2. `migrations/Notebooks/NB_[NOMBRE_JOB].py`
3. `migrations/SQL/NB_[NOMBRE_JOB]_*.sql`
4. Actualizar: `constants.env` y `ds_catalog.txt` (si aplica)

**PROCESO:**
- FASE 1: Análisis XML (Extracción Semántica Selectiva - ver 02_analysis_rules.md)
- FASE 2: Generación de artefactos (ver 04, 05, 07_*_rules.md)
- FASE 3: Validación (ejecutar checklist de 06_migration_process_rules.md)

**REGLAS CRÍTICAS:**
⚠️ Las 3 más importantes (el resto está en archivos de reglas):
1. NO simplificar análisis por "límite de tokens"
2. Diagrama DEBE usar sintaxis ```mermaid (NO ASCII)
3. SQL en archivos .sql separados (NO inline en notebooks)
4. Archivo análisis DEBE incluir sección "Equivalencias de Datasets para ds_catalog.txt"
5. Transformers con múltiples salidas: COPIAR constraints literales (preservar OR/AND)

**ANTI-PATTERNS PROHIBIDOS:**
❌ from startup import | ❌ add_widget() | ❌ conf.get_param() | ❌ df= o datos= 
❌ SQL inline f\"SELECT | ❌ UNION inline | ❌ {PARAM} en SQL (usar #PARAM#)

**VALIDACIÓN PRE-REPORTE:**
Antes de reportar completitud, verificar físicamente:
- [ ] 3 archivos creados (MD, PY, SQL)
- [ ] Diagrama en ```mermaid
- [ ] NO patrones prohibidos en PY
- [ ] Parámetros SQL usan #PARAM#
- [ ] ds_catalog.txt actualizado (si hay datasets nuevos)

**REPORTE ESPERADO:**
✅ Artefactos creados: [lista]
⚠️ Advertencias: [lista o "ninguna"]
❌ Errores: [lista o "ninguno"]

Contexto de archivos: 
.github/instructions/parallel_rules/06_migration_process_rules.md (PRINCIPAL),
.github/instructions/parallel_rules/02_analysis_rules.md,
.github/instructions/parallel_rules/04_notebook_rules.md,
.github/instructions/parallel_rules/05_sql_rules.md,
.github/instructions/parallel_rules/07_dataset_rules.md"
```

#### 📏 Métricas de Mejora
- **Antes:** 152 líneas de instrucciones
- **Después:** ~50 líneas de instrucciones
- **Reducción:** 67% menos contenido
- **Beneficio:** Foco en lo crítico, mejor mantenibilidad, menor costo de tokens

#### 🔧 Pasos de Implementación
1. Hacer backup del archivo actual
2. Reemplazar sección líneas 85-237
3. Validar que 06_migration_process_rules.md contiene TODAS las reglas detalladas
4. Probar con 1 job de ejemplo
5. Si funciona correctamente, proceder con el resto

---

## 🟡 PRIORIDAD MEDIA

### 2. REPETICIÓN Y DUPLICACIÓN DE REGLAS

#### 📍 Ubicación Principal
**Archivos afectados:**
1. `.github/agents/job-ds-migration.agent.md` (líneas 85-237)
2. `.github/agents/workflow-ds-migration.agent.md` (líneas 26-29)
3. `.github/instructions/parallel_rules/*.md` (fuente de verdad)

#### ❌ Problema Actual
Las mismas reglas aparecen en **múltiples lugares**:

**Ejemplo 1: Reglas de Mermaid**
- Aparece en: `job-ds-migration.agent.md` línea 186
- Aparece en: `workflow-ds-migration.agent.md` línea 28  
- Definida en: `02_analysis_rules.md` (fuente original)

**Ejemplo 2: Reglas de Anti-Patterns**
- Listadas completamente en: `job-ds-migration.agent.md` líneas 135-145
- Definidas detalladamente en: `04_notebook_rules.md`

**Consecuencias:**
- Si se actualiza una regla, hay que cambiarla en 3+ lugares
- Riesgo de inconsistencias entre versiones
- Violación del principio DRY (Don't Repeat Yourself)

#### ✅ Solución Propuesta

**Estrategia: Fuente Única de Verdad**

1. **Mantener reglas SOLO en:**
   - `.github/instructions/parallel_rules/*.md` (nivel detallado)
   - `.github/instructions/sequence_rules/*.md` (nivel detallado)

2. **En agentes: Solo referenciar, no copiar**

**Modificación en `.github/agents/job-ds-migration.agent.md`:**

**Líneas 85-237 - Reemplazar con:**
```markdown
### 3. Delegación por Job

Para cada Job del inventario, delegar usando este template:

**TEMPLATE DE DELEGACIÓN:**
```
#runSubagent "MIGRACIÓN JOB: [NOMBRE_JOB_XML]

Ver archivo de contexto: .github/instructions/parallel_rules/06_migration_process_rules.md
Este archivo contiene el proceso completo y referencias a todas las reglas necesarias.

ARTEFACTOS ESPERADOS:
- migrations/XML_ANALISIS/[NOMBRE]_ANALYSIS.md
- migrations/Notebooks/NB_[NOMBRE].py  
- migrations/SQL/NB_[NOMBRE]_*.sql

Ver: .github/instructions/parallel_rules/06_migration_process_rules.md"
```

**Variables a reemplazar:**
- `[NOMBRE_JOB_XML]`: Nombre completo del XML
```

**Modificación en `.github/agents/workflow-ds-migration.agent.md`:**

**Líneas 26-29 - Cambiar de:**
```markdown
2. `.github/instructions/sequence_rules/02_analysis_rules.md` - Estructura del análisis (8 secciones obligatorias)
3. `.github/instructions/sequence_rules/03_notebook_rules.md` - Notebooks decisores (referencia)
4. `.github/instructions/sequence_rules/04_workflow_rules.md` - Patrones de workflows YAML (5 patrones padre + 5 hijo)
```

**A:**
```markdown
2-4. Ver detalles completos en: `.github/instructions/sequence_rules/01_rules.md`
    (Este archivo contiene referencias a todos los archivos de reglas específicas)
```

#### 📏 Métricas de Mejora
- **Lugares donde aparecen reglas duplicadas:** 3+ archivos
- **Después:** 1 solo lugar (archivos de instrucciones)
- **Beneficio:** Actualizaciones 3x más rápidas, sin inconsistencias

---

### 3. PROMPTS SIMPLES SUBDESARROLLADOS

#### 📍 Ubicación
**Archivo 1:** `.github/prompts/migrar-jobs.prompt.md`  
**Líneas:** 1-7 (TODO el archivo)

**Archivo 2:** `.github/prompts/create-cleaner.prompt.md`  
**Líneas:** 1-7 (TODO el archivo)

#### ❌ Problema Actual

**migrar-jobs.prompt.md (contenido actual):**
```markdown
---
description: Migrate jobs from IBM DataStage to Databricks.
name: "migrar-jobs"
---
 Realiza la migración de DataStage a Databricks de todos y cada uno de los Jobs siguiendo las reglas.
```

**Problemas identificados:**
1. **Demasiado genérico** - No especifica qué hacer primero
2. **No indica cuál agente invocar** - Usuario queda sin dirección
3. **No maneja escenarios** - ¿Migración inicial? ¿Continuación? ¿Corrección?
4. **No da contexto** - ¿Dónde están las reglas?
5. **Prompt de 1 línea** - Insuficiente para guiar al agente correctamente

**create-cleaner.prompt.md (contenido actual):**
```markdown
---
description:  Crear el notebook Cleaner.
name: "create-cleaner"
---
 Crea el notebook cleaner de borrado de tablas delta basado en las reglas.
```

**Problemas similares:**
- No especifica QUÉ notebooks escanear
- No indica el PATRÓN de nombrado esperado
- Falta contexto sobre el objetivo del cleaner

#### ✅ Solución Propuesta

**REEMPLAZAR COMPLETAMENTE `.github/prompts/migrar-jobs.prompt.md`:**

```markdown
---
description: Migrar jobs paralelos de IBM DataStage a Databricks
name: "migrar-jobs"
---

# Migración de Jobs DataStage → Databricks

ANTES de iniciar, identifica el ESCENARIO:

## 🔍 Identificación de Escenario

**Pregúntate:**
1. ¿El usuario quiere migrar TODOS los jobs por primera vez?
   → **Escenario 1: Migración Inicial Completa**

2. ¿El usuario quiere CONTINUAR una migración ya iniciada?
   → **Escenario 2: Continuación de Migración Interrumpida**
   → Verificar: ¿Qué jobs ya tienen artefactos en `migrations/`?

3. ¿El usuario menciona jobs ESPECÍFICOS para revisar/corregir?
   → **Escenario 3: Corrección de Jobs Específicos**
   → Identificar: ¿Cuáles jobs menciona?

4. ¿Es una consulta general SIN pedir migración?
   → **Escenario 4: Consulta No-Migratoria**
   → Responder directamente, NO invocar agente

## 🎯 Acción

**Para Escenarios 1, 2 o 3:**
Invoca al agente especializado: **@job-ds-migration-agent**

**Para Escenario 4:**
Responde la consulta directamente sin invocar agentes.

## 📚 Contexto

Los jobs están en: `DATASTAGE/Parallel Jobs/*.xml`
Las reglas están en: `.github/instructions/parallel_rules/`
El agente orquestador creará subagentes para cada job individual.
```

**REEMPLAZAR COMPLETAMENTE `.github/prompts/create-cleaner.prompt.md`:**

```markdown
---
description: Crear notebook de limpieza centralizado para tablas delta temporales
name: "create-cleaner"
---

# Creación de Notebook Cleaner

## 🎯 Objetivo
Crear UN SOLO notebook que elimine TODAS las tablas delta temporales generadas por los notebooks de jobs migrados.

## 🔍 Proceso

1. **Escanear notebooks migrados:**
   - Ubicación: `DATABRICKS/migrations/Notebooks/NB_*.py`
   - Excluir: `*_CLEANER.py` (cleaners existentes)

2. **Extraer tablas delta:**
   - Buscar patrón: `db.write_delta(...)`
   - Identificar nombres de tablas y sufijos dinámicos

3. **Identificar prefijo común:**
   - Encontrar el prefijo compartido entre todos los notebooks
   - Ejemplo: `NB_JP_PATRIF_` es prefijo de `NB_JP_PATRIF_0800_*`, `NB_JP_PATRIF_GMO_*`

4. **Generar cleaner:**
   - Nombre: `[PREFIJO_COMÚN]_CLEANER.py`
   - Ubicación: `DATABRICKS/migrations/Notebooks/`
   - Estructura: 7 celdas con organización por notebook origen

## 🤖 Invocación

Invoca al agente especializado: **@nb-cleaner-creator**

## 📚 Reglas Completas

Ver: `.github/instructions/parallel_rules/08_delta_cleaner_rules.md`
```

#### 📏 Métricas de Mejora
- **migrar-jobs.prompt.md:**
  - Antes: 1 línea de instrucciones
  - Después: ~30 líneas con lógica de escenarios
  - Mejora: 30x más contexto
  
- **create-cleaner.prompt.md:**
  - Antes: 1 línea de instrucciones  
  - Después: ~25 líneas con proceso detallado
  - Mejora: 25x más claridad

---

### 4. MANEJO DE ERRORES INSUFICIENTE

#### 📍 Ubicación
**Archivo:** `.github/agents/job-ds-migration.agent.md`  
**Sección faltante:** Entre "Validación Post-Ejecución" (línea 269) y "Criterio de Continuación" (línea 282)  
**Problema:** No existe una sección dedicada al manejo de errores

#### ❌ Problema Actual

**Situaciones sin estrategia definida:**

1. **¿Qué pasa si un subagente falla?**
   - Actualmente: No hay instrucciones claras
   - Consecuencia: El orquestador podría detenerse completamente

2. **¿Qué pasa si falta un archivo XML?**
   - Actualmente: No documentado
   - Consecuencia: Error inesperado

3. **¿Qué pasa si un job genera error parcial?** (ej: 2 de 3 artefactos)
   - Actualmente: No hay criterio de "job completado vs fallido"
   - Consecuencia: Estado inconsistente

4. **¿Cómo recuperarse de una sesión interrumpida?**
   - Actualmente: Escenario 2 asume que jobs "completos" tienen todos los artefactos
   - Problema: ¿Qué pasa con jobs a medio hacer?

**Búsqueda de "error" en el archivo:**
```bash
# Resultado: Solo aparece en "Jobs con Errores" (línea 299)
# No hay estrategia de manejo
```

#### ✅ Solución Propuesta

**INSERTAR nueva sección después de línea 269:**

```markdown
### 4. Estrategia de Manejo de Errores

#### Política de Continuidad
**REGLA:** Un error en un job individual NO debe detener el proceso completo.

#### Categorías de Error

##### A. Error Crítico del Subagente
**Síntomas:**
- Subagente reporta "ERROR CRÍTICO"
- Subagente no puede completar análisis del XML
- Subagente se detiene antes de generar archivos

**Acción:**
1. Registrar el job como "FALLIDO" con detalle del error
2. NO intentar reintento automático (podría repetir el mismo error)
3. Continuar con el siguiente job del inventario
4. Al final del proceso, reportar lista de jobs fallidos

**Registro:**
```
❌ JOB FALLIDO: [NOMBRE_JOB]
   Razón: [Descripción del error del subagente]
   Puede reintentarse manualmente con: @job-ds-migration-agent corrección [NOMBRE_JOB]
```

##### B. Artefactos Incompletos
**Síntomas:**
- Subagente reporta completitud PERO verificación física muestra archivos faltantes
- Ejemplo: existe MD y PY pero faltan archivos SQL

**Acción:**
1. Registrar job como "INCOMPLETO"
2. Listar artefactos faltantes específicos
3. Marcar para revisión manual
4. Continuar con siguiente job

**Registro:**
```
⚠️ JOB INCOMPLETO: [NOMBRE_JOB]
   Artefactos creados: [MD ✅, PY ✅]
   Artefactos faltantes: [SQL/*.sql ❌]
   Requiere revisión manual
```

##### C. Archivo XML No Encontrado
**Síntomas:**
- El archivo `DATASTAGE/Parallel Jobs/[NOMBRE].xml` no existe
- Puede ocurrir si el usuario especifica un nombre incorrecto

**Acción:**
1. Reportar INMEDIATAMENTE sin invocar subagente
2. Sugerir archivos similares (búsqueda difusa)
3. NO continuar con ese job
4. Continuar con siguiente job si hay más en la cola

**Registro:**
```
❌ XML NO ENCONTRADO: [NOMBRE_JOB].xml
   Ubicación esperada: DATASTAGE/Parallel Jobs/[NOMBRE_JOB].xml
   ¿Quisiste decir?: [sugerencias de archivos similares]
```

##### D. Error de Validación Post-Ejecución
**Síntomas:**
- Archivos existen pero contienen anti-patterns
- Ejemplo: notebook contiene `from startup import` o SQL usa `{PARAM}`

**Acción:**
1. Reportar como "COMPLETADO CON ADVERTENCIAS"
2. Listar violaciones específicas encontradas
3. El job se considera técnicamente completo (para inventario)
4. Marcar para revisión de calidad

**Registro:**
```
⚠️ COMPLETADO CON ADVERTENCIAS: [NOMBRE_JOB]
   Artefactos: Todos generados ✅
   Advertencias de calidad:
   - Notebook contiene anti-pattern: df= en línea 45
   - SQL usa {PARAM} en lugar de #PARAM#
   Requiere corrección de calidad
```

#### Registro de Estado

**Mantener durante ejecución:**
```python
Estado del Proceso:
├─ Total de Jobs: [N]
├─ Completados: [X] ✅
├─ Fallidos: [Y] ❌
├─ Incompletos: [Z] ⚠️
└─ Pendientes: [N-X-Y-Z] ⏳
```

#### Recuperación de Sesión Interrumpida

**Si el usuario debe interrumpir:**
1. El progreso está registrado en artefactos físicos
2. Al reiniciar (Escenario 2), verificar:
   - Jobs CON artefactos completos → Saltar
   - Jobs SIN artefactos o con artefactos parciales → Reintentar
3. Mantener logs de jobs problemáticos para evitar reintentos infinitos

#### Escalación

**Cuándo escalar al usuario:**
- 3+ jobs consecutivos fallan con el mismo error → Problema sistémico
- 50%+ de jobs del inventario fallan → Revisar configuración
- Error en archivos de reglas (no existen o tienen sintaxis incorrecta)

**Mensaje de escalación:**
```
🚨 PROBLEMA SISTÉMICO DETECTADO

[X] jobs consecutivos han fallado con error similar:
[Descripción del patrón de error]

Recomendación:
- Revisar configuración del proyecto
- Verificar que archivos de reglas existan y sean accesibles
- Validar estructura de carpetas DATASTAGE/ y DATABRICKS/

¿Deseas continuar con los jobs restantes o detener para investigar?
```
```

#### 📏 Impacto de la Mejora
- **Antes:** Sin estrategia → proceso frágil
- **Después:** Manejo robusto → proceso resiliente
- **Beneficio:** Migración de 100 jobs no se detiene por 1 error

---

## 🟢 PRIORIDAD BAJA

### 5. INCONSISTENCIAS EN METADATA DEL FRONT-MATTER

#### 📍 Ubicaciones

**Archivo 1:** `.github/agents/workflow-ds-migration.agent.md`  
**Líneas:** 1-10 (front-matter)  
**Problema:** Falta definición de `handoffs`

**Archivo 2:** `.github/agents/nb-cleaner-creator.agent.md`  
**Líneas:** 13-17 (handoffs)  
**Problema:** Handoff unidireccional (apunta a workflow pero workflow no apunta de vuelta)

**Archivo 3:** `.github/agents/job-ds-migration.agent.md`  
**Líneas:** 1-11 (front-matter)  
**Observación:** Este SÍ tiene handoffs correctamente definidos (líneas 6-10)

#### ❌ Problema Actual

**workflow-ds-migration.agent.md (líneas 1-10):**
```yaml
---
name: "workflow-ds-migration-agent"
description: "Agente encargado de crear Pipeline de Databricks"
tools:
  - "execute"
  - "read"
  - "edit"
  - "search"
  - "agent"
  - "todo"
model: "Claude Sonnet 4.5 (copilot)"

---
```
❌ **Falta:** Sección `handoffs` para navegación a otros agentes

**nb-cleaner-creator.agent.md (líneas 13-17):**
```yaml
handoffs:
  - label: "Crea Pipeline de Databricks"
    agent: "workflow-ds-migration-agent"
    prompt: "Crea el pipeline de Databricks de los Jobs migrados"
    send: true
```
✅ Correcto PERO...  
❌ **Problema:** `workflow-ds-migration-agent` no tiene handoff de vuelta

**Consecuencia:**
- Navegación inconsistente entre agentes
- Flujo de trabajo incompleto
- Usuario no puede moverse fácilmente entre pasos relacionados

#### ✅ Solución Propuesta

**MODIFICAR `.github/agents/workflow-ds-migration.agent.md` líneas 1-10:**

**Cambiar de:**
```yaml
---
name: "workflow-ds-migration-agent"
description: "Agente encargado de crear Pipeline de Databricks"
tools:
  - "execute"
  - "read"
  - "edit"
  - "search"
  - "agent"
  - "todo"
model: "Claude Sonnet 4.5 (copilot)"

---
```

**A:**
```yaml
---
name: "workflow-ds-migration-agent"
description: "Agente encargado de crear Pipeline de Databricks (workflows YAML)"
tools:
  - "execute"
  - "read"
  - "edit"
  - "search"
  - "agent"
  - "todo"
model: "Claude Sonnet 4.5 (copilot)"
handoffs:
  - label: "Crear notebook cleaner"
    agent: "nb-cleaner-creator"
    prompt: "Crea el notebook cleaner de limpieza de tablas delta para los jobs migrados"
    send: true
  - label: "Migrar jobs paralelos"
    agent: "job-ds-migration-agent"
    prompt: "Inicia o continúa la migración de jobs de DataStage a Databricks"
    send: false

---
```

**Justificación de cambios:**
1. ✅ Agregado handoff a `nb-cleaner-creator` (navegación bidireccional)
2. ✅ Agregado handoff a `job-ds-migration-agent` (para volver al inicio del flujo)
3. ✅ `send: true` en cleaner (envía contexto actual)
4. ✅ `send: false` en job-migration (nuevo contexto, no heredar)

**OPCIONAL - Mejorar simetría en `nb-cleaner-creator.agent.md` líneas 13-17:**

**Cambiar de:**
```yaml
handoffs:
  - label: "Crea Pipeline de Databricks"
    agent: "workflow-ds-migration-agent"
    prompt: "Crea el pipeline de Databricks de los Jobs migrados"
    send: true
```

**A:**
```yaml
handoffs:
  - label: "Crear workflow de Databricks"
    agent: "workflow-ds-migration-agent"
    prompt: "Crea los workflows YAML de orquestación para los jobs migrados"
    send: true
  - label: "Migrar más jobs"
    agent: "job-ds-migration-agent"
    prompt: "Continuar con la migración de jobs adicionales"
    send: false
```

#### 📏 Métricas de Mejora
- **Antes:** 1 agente sin handoffs, flujo unidireccional
- **Después:** 3 agentes totalmente conectados, flujo bidireccional
- **Beneficio:** Navegación intuitiva, mejor UX

---

## 📊 RESUMEN DE ARCHIVOS A MODIFICAR

| # | Archivo | Líneas Afectadas | Tipo de Cambio | Prioridad |
|---|---------|------------------|----------------|-----------|
| 1 | `.github/agents/job-ds-migration.agent.md` | 85-237 | Condensar template | 🔴 Alta |
| 2 | `.github/agents/job-ds-migration.agent.md` | +270 (insertar) | Agregar sección errores | 🟡 Media |
| 3 | `.github/prompts/migrar-jobs.prompt.md` | 1-7 (reemplazar todo) | Expandir contenido | 🟡 Media |
| 4 | `.github/prompts/create-cleaner.prompt.md` | 1-7 (reemplazar todo) | Expandir contenido | 🟡 Media |
| 5 | `.github/agents/workflow-ds-migration.agent.md` | 1-10 | Agregar handoffs | 🟢 Baja |
| 6 | `.github/agents/nb-cleaner-creator.agent.md` | 13-17 | Mejorar handoffs | 🟢 Baja |

**Total de modificaciones:** 6 cambios en 4 archivos

---

## 🎯 PLAN DE IMPLEMENTACIÓN SUGERIDO

### Fase 1: Cambios Críticos (Semana 1)
**Objetivo:** Mejorar eficiencia y reducir costos

- [ ] **Día 1-2:** Implementar cambio #1 (condensar template de delegación)
  - Hacer backup de `job-ds-migration.agent.md`
  - Aplicar nueva versión del template
  - Probar con 2-3 jobs de ejemplo
  - Validar que resultados son idénticos
  
- [ ] **Día 3:** Validar que las reglas en archivos son suficientes
  - Revisar `06_migration_process_rules.md`
  - Confirmar que contiene TODAS las instrucciones necesarias
  - Si falta algo, agregar a ese archivo (NO al agente)

### Fase 2: Robustez (Semana 2)
**Objetivo:** Hacer el sistema más resiliente

- [ ] **Día 4-5:** Implementar cambio #2 (manejo de errores)
  - Insertar nueva sección en `job-ds-migration.agent.md`
  - Probar con un job que deliberadamente falle
  - Validar que el proceso continúa con el siguiente job

- [ ] **Día 6:** Implementar cambios #3 y #4 (mejorar prompts)
  - Reemplazar contenido de `migrar-jobs.prompt.md`
  - Reemplazar contenido de `create-cleaner.prompt.md`
  - Probar invocación desde interfaz de usuario

### Fase 3: Pulido (Semana 3)
**Objetivo:** Mejorar experiencia de usuario

- [ ] **Día 7:** Implementar cambios #5 y #6 (handoffs)
  - Agregar handoffs a `workflow-ds-migration.agent.md`
  - Mejorar handoffs en `nb-cleaner-creator.agent.md`
  - Probar navegación entre agentes

- [ ] **Día 8:** Validación end-to-end
  - Migrar 5 jobs reales usando el sistema mejorado
  - Documentar cualquier problema encontrado
  - Ajustar si es necesario

### Fase 4: Documentación (Semana 4)
- [ ] **Día 9:** Actualizar documentación del proyecto
- [ ] **Día 10:** Crear guía de troubleshooting basada en manejo de errores

---

## 📈 MÉTRICAS DE ÉXITO

### KPIs a Medir

| Métrica | Antes | Después (Objetivo) |
|---------|-------|-------------------|
| Líneas de instrucciones por delegación | 152 | <60 |
| Tokens promedio por job | ~8,000 | <3,000 |
| Tasa de éxito en migración con errores | Desconocida | >95% continuidad |
| Tiempo para identificar escenario | Manual | <30 segundos |
| Facilidad de navegación entre agentes | Limitada | Bidireccional completa |

### Validaciones de Calidad
- [ ] 10 jobs migrados sin errores con el sistema mejorado
- [ ] 2 jobs con error deliberado NO detienen el proceso
- [ ] Usuario puede navegar entre los 3 agentes sin confusión
- [ ] Prompts simples invocan correctamente a los agentes

---

## 🔗 REFERENCIAS

### Archivos Analizados
1. `.github/agents/job-ds-migration.agent.md` (347 líneas)
2. `.github/agents/workflow-ds-migration.agent.md` (102 líneas estimadas)
3. `.github/agents/nb-cleaner-creator.agent.md` (259 líneas)
4. `.github/prompts/migrar-jobs.prompt.md` (7 líneas)
5. `.github/prompts/create-cleaner.prompt.md` (7 líneas)

### Reglas Relacionadas
- `.github/instructions/parallel_rules/01_rules.md` - Estructura del proyecto
- `.github/instructions/parallel_rules/02_analysis_rules.md` - Análisis de XML
- `.github/instructions/parallel_rules/04_notebook_rules.md` - Creación de notebooks
- `.github/instructions/parallel_rules/06_migration_process_rules.md` - Proceso completo
- `.github/instructions/sequence_rules/*.md` - Reglas de workflows

---

## ✅ CHECKLIST DE IMPLEMENTACIÓN

### Antes de Empezar
- [ ] Hacer backup de todos los archivos a modificar
- [ ] Crear branch de Git para cambios: `feature/mejora-agentes-copilot`
- [ ] Leer este documento completamente

### Durante Implementación
- [ ] Aplicar cambios en orden de prioridad (Alta → Media → Baja)
- [ ] Probar cada cambio individual antes de continuar
- [ ] Documentar cualquier desviación del plan

### Después de Implementar
- [ ] Validar con 5+ jobs de ejemplo
- [ ] Medir KPIs documentados
- [ ] Crear pull request con descripción detallada
- [ ] Actualizar este documento con resultados reales

---

## 📝 NOTAS ADICIONALES

### Consideraciones de Compatibilidad
- Todos los cambios son **backwards compatible** con XMLs existentes
- Los artefactos ya generados NO necesitan regenerarse
- Los subagentes existentes funcionarán con el nuevo template condensado

### Rollback Plan
Si algo falla después de implementar:
1. Restaurar desde backup
2. Identificar el cambio específico que causó el problema
3. Implementar ese cambio de forma aislada
4. Investigar y ajustar antes de reintegrarlo

### Próximos Pasos Futuros
(No incluidos en este documento, para consideración futura)
- Agregar métricas automáticas de calidad
- Implementar tests automatizados para agentes
- Crear dashboard de progreso de migración
- Agregar logging estructurado de todas las operaciones

---

**Documento preparado por:** Análisis de GitHub Copilot Agents  
**Fecha:** 19 de Febrero, 2026  
**Versión:** 1.0  
**Estado:** ✅ Listo para Revisión e Implementación
