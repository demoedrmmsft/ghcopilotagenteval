# Evaluación Manual - @datastageagent
Fecha: 2026-02-19

## Modelos evaluados
1. GPT-5.3-Codex
2. Claude Sonnet 4.6
3. Claude Opus 4.6

## Escala de calificación
- ✅ 3 = Excelente (código correcto, completo, bien explicado)
- ⚠️ 2 = Aceptable (código presente pero incompleto o con errores menores)
- ❌ 1 = Deficiente (sin código, respuesta incorrecta o muy vaga)

---

## Test 1 — Traducción de expresión simple
**Prompt**: `Traduce esta expresión de DataStage a PySpark: Trim(Upcase(FirstName)) : " " : Trim(Upcase(LastName))`

| Criterio | GPT-5.3-Codex | Claude Sonnet 4.6 | Claude Opus 4.6 |
|----------|:---:|:---:|:---:|
| Código correcto (F.concat, F.trim, F.upper) | _ | _ | _ |
| Explicación clara | _ | _ | _ |
| Muestra DataStage vs PySpark lado a lado | _ | _ | _ |
| **TOTAL (max 9)** | _ | _ | _ |

**Notas**:
- GPT-5.3-Codex:
- Claude Sonnet 4.6:
- Claude Opus 4.6:

---

## Test 2 — Traducción de expresión con lógica condicional
**Prompt**: `Traduce esta expresión de DataStage a PySpark: If IsNull(Status) Then "UNKNOWN" Else Upcase(Trim(Status))`

| Criterio | GPT-5.3-Codex | Claude Sonnet 4.6 | Claude Opus 4.6 |
|----------|:---:|:---:|:---:|
| Usa F.when().otherwise() correctamente | _ | _ | _ |
| Maneja el caso nulo bien | _ | _ | _ |
| Código ejecutable sin errores | _ | _ | _ |
| **TOTAL (max 9)** | _ | _ | _ |

**Notas**:
- GPT-5.3-Codex:
- Claude Sonnet 4.6:
- Claude Opus 4.6:

---

## Test 3 — Migración completa de job DSX
**Prompt**: `@datastageagent migra el archivo test-artifacts/01_simple_customer_etl.dsx a Databricks`

| Criterio | GPT-5.3-Codex | Claude Sonnet 4.6 | Claude Opus 4.6 |
|----------|:---:|:---:|:---:|
| Lee el archivo DSX correctamente | _ | _ | _ |
| Genera notebook con todas las secciones (params, read, transform, write) | _ | _ | _ |
| Traduce las 7 derivaciones del Transformer | _ | _ | _ |
| Aplica los 3 constraints | _ | _ | _ |
| Usa Delta Lake (no CSV) | _ | _ | _ |
| Incluye validación post-escritura | _ | _ | _ |
| **TOTAL (max 18)** | _ | _ | _ |

**Notas**:
- GPT-5.3-Codex:
- Claude Sonnet 4.6:
- Claude Opus 4.6:

---

## Test 4 — Explicación de componente
**Prompt**: `Explica cómo migrar un Aggregator stage de DataStage a PySpark con un ejemplo`

| Criterio | GPT-5.3-Codex | Claude Sonnet 4.6 | Claude Opus 4.6 |
|----------|:---:|:---:|:---:|
| Explica el propósito del Aggregator | _ | _ | _ |
| Muestra groupBy().agg() con ejemplo | _ | _ | _ |
| Incluye varias funciones (sum, count, max) | _ | _ | _ |
| **TOTAL (max 9)** | _ | _ | _ |

**Notas**:
- GPT-5.3-Codex:
- Claude Sonnet 4.6:
- Claude Opus 4.6:

---

## Resumen Final

| Modelo | Test 1 | Test 2 | Test 3 | Test 4 | TOTAL | Velocidad |
|--------|:------:|:------:|:------:|:------:|:-----:|:---------:|
| GPT-5.3-Codex | _ | _ | _ | _ | _ | _ |
| Claude Sonnet 4.6 | _ | _ | _ | _ | _ | _ |
| Claude Opus 4.6 | _ | _ | _ | _ | _ | _ |

**Máximo posible**: 45 puntos

## Conclusión
- **Mejor modelo general**: _____________
- **Mejor para migraciones completas**: _____________
- **Mejor balance calidad/velocidad**: _____________
- **Modelo recomendado para uso diario**: _____________
