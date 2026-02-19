# Guía de Evaluación del Agente DataStage

## Opciones para Evaluar el Agente con Diferentes Modelos

### 📊 Resumen Ejecutivo

Para evaluar tu agente `@datastageagent` con diferentes modelos (Claude Sonnet, GPT-4, etc.), tienes **3 opciones principales**:

| Opción | Dónde Corre | Modelos Disponibles | Complejidad | Mejor Para |
|--------|-------------|---------------------|-------------|------------|
| **1. Azure AI Foundry** | Cloud (Azure) | GPT-4, GPT-4o, Claude, Mistral, Llama | Media | **✅ RECOMENDADO** - Producción, múltiples modelos, métricas avanzadas |
| **2. GitHub Models** | Cloud (GitHub) | GPT-4o, Claude Sonnet, Phi, Llama | Baja | Pruebas rápidas, prototipado |
| **3. Local con AI Toolkit** | Local (VS Code) | Modelos locales + API externa | Alta | Control total, evaluación personalizada |

---

## Opción 1: Azure AI Foundry (RECOMENDADO) 🏆

### ¿Por qué Azure AI Foundry?

- ✅ **Múltiples modelos en un solo lugar**: GPT-4, Claude Sonnet, Mistral, Llama
- ✅ **Métricas integradas**: Groundedness, Relevance, Coherence, Fluency
- ✅ **Evaluadores personalizados**: Puedes crear tus propias métricas
- ✅ **Comparación lado a lado**: Compara resultados de diferentes modelos
- ✅ **Logging automático**: Rastrea todas las ejecuciones y métricas
- ✅ **Integración con código**: Ejecuta desde tu proyecto local

### Configuración

#### 1. Instalar dependencias

```bash
pip install azure-ai-projects azure-identity promptflow
```

#### 2. Crear proyecto en Azure AI Foundry

```bash
# Opción A: Desde portal Azure
# https://ai.azure.com → Create new project

# Opción B: Desde CLI
az ml workspace create --name datastage-eval --resource-group my-rg
```

#### 3. Configurar conexión

```python
# config.py
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

project_client = AIProjectClient(
    subscription_id="YOUR_SUBSCRIPTION_ID",
    resource_group_name="YOUR_RESOURCE_GROUP",
    project_name="datastage-eval",
    credential=DefaultAzureCredential()
)
```

### Script de Evaluación

```python
# evaluate_agent.py
"""
Evalúa el agente DataStage con múltiples modelos usando Azure AI Foundry
"""
import json
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    EvaluationConfig,
    EvaluationRun,
    ModelConfiguration
)
from azure.identity import DefaultAzureCredential

# Configuración del proyecto
client = AIProjectClient(
    subscription_id="YOUR_SUBSCRIPTION_ID",
    resource_group_name="YOUR_RESOURCE_GROUP", 
    project_name="datastage-eval",
    credential=DefaultAzureCredential()
)

# Dataset de evaluación: casos de prueba para el agente
test_queries = [
    {
        "id": "test_001",
        "query": "Traduce esta expresión de DataStage a PySpark: Trim(Upcase(FirstName)) : \" \" : Trim(Upcase(LastName))",
        "expected_keywords": ["F.concat", "F.trim", "F.upper"],
        "category": "expression_translation"
    },
    {
        "id": "test_002", 
        "query": "Migra el archivo test-artifacts/01_simple_customer_etl.dsx a Databricks",
        "expected_keywords": ["spark.read", "withColumn", "Delta Lake", "transform"],
        "category": "full_migration"
    },
    {
        "id": "test_003",
        "query": "Cómo implemento SCD Type 2 en Databricks?",
        "expected_keywords": ["MERGE", "effective_date", "is_current", "window"],
        "category": "pattern_explanation"
    },
    {
        "id": "test_004",
        "query": "Qué hace un Aggregator stage en DataStage?",
        "expected_keywords": ["groupBy", "agg", "sum", "count", "max"],
        "category": "component_explanation"
    }
]

# Modelos a evaluar
models_to_test = [
    {
        "name": "gpt-4o",
        "deployment": "gpt-4o",  # Nombre del deployment en Azure
        "provider": "azure_openai"
    },
    {
        "name": "claude-sonnet-3.5",
        "deployment": "claude-sonnet-35",
        "provider": "azure_ai"
    },
    {
        "name": "gpt-4-turbo",
        "deployment": "gpt-4-turbo",
        "provider": "azure_openai"
    }
]

# Métricas de evaluación personalizadas para el agente
def evaluate_response(query, response, expected_keywords, category):
    """
    Evaluador personalizado para el agente DataStage
    """
    scores = {}
    
    # 1. Completeness: ¿Contiene las palabras clave esperadas?
    keywords_found = sum(1 for kw in expected_keywords if kw.lower() in response.lower())
    scores['keyword_coverage'] = keywords_found / len(expected_keywords) if expected_keywords else 0
    
    # 2. Code Quality: ¿Incluye código ejecutable?
    has_code = any(marker in response for marker in ['```python', 'F.', 'spark.', 'df.'])
    scores['includes_code'] = 1.0 if has_code else 0.0
    
    # 3. Structure: ¿Tiene buena estructura?
    has_headers = any(marker in response for marker in ['##', '###', '**'])
    has_explanation = len(response) > 200
    scores['well_structured'] = 1.0 if (has_headers and has_explanation) else 0.5 if has_headers else 0.0
    
    # 4. Specific to category
    if category == "expression_translation":
        # Debe incluir tanto la expresión DataStage como PySpark
        has_comparison = "DataStage" in response and "PySpark" in response
        scores['category_specific'] = 1.0 if has_comparison else 0.0
        
    elif category == "full_migration":
        # Debe incluir múltiples etapas del pipeline
        stages = ["read", "transform", "write", "validate"]
        stages_mentioned = sum(1 for s in stages if s.lower() in response.lower())
        scores['category_specific'] = stages_mentioned / len(stages)
        
    elif category == "pattern_explanation":
        # Debe incluir código de ejemplo y explicación
        has_example = "```" in response
        has_description = len(response) > 300
        scores['category_specific'] = 1.0 if (has_example and has_description) else 0.5
        
    else:  # component_explanation
        # Debe explicar qué hace y mostrar equivalente PySpark
        has_purpose = any(word in response.lower() for word in ["purpose", "propósito", "qué hace"])
        has_equivalent = "pyspark" in response.lower() or "spark" in response.lower()
        scores['category_specific'] = 1.0 if (has_purpose and has_equivalent) else 0.5
    
    # Score general
    scores['overall'] = sum(scores.values()) / len(scores)
    
    return scores

# Función para ejecutar la evaluación
def run_evaluation():
    results = {}
    
    for model_config in models_to_test:
        print(f"\n{'='*60}")
        print(f"Evaluando modelo: {model_config['name']}")
        print(f"{'='*60}\n")
        
        model_results = []
        
        for test in test_queries:
            print(f"  Procesando: {test['id']} - {test['category']}")
            
            # Aquí harías la llamada al modelo
            # Por ahora simulamos la respuesta (en producción usarías el cliente de Azure AI)
            """
            response = client.chat.completions.create(
                model=model_config['deployment'],
                messages=[
                    {"role": "system", "content": "You are @datastageagent, an expert in DataStage to Databricks migration."},
                    {"role": "user", "content": test['query']}
                ],
                temperature=0.7
            )
            response_text = response.choices[0].message.content
            """
            
            # Simulación para este ejemplo
            response_text = f"[Respuesta del modelo {model_config['name']} para {test['query']}]"
            
            # Evaluar la respuesta
            scores = evaluate_response(
                test['query'],
                response_text,
                test['expected_keywords'],
                test['category']
            )
            
            result = {
                "test_id": test['id'],
                "category": test['category'],
                "query": test['query'],
                "response": response_text,
                "scores": scores
            }
            
            model_results.append(result)
            
            print(f"    ✅ Overall Score: {scores['overall']:.2%}")
        
        results[model_config['name']] = model_results
    
    return results

# Generar reporte comparativo
def generate_report(results):
    """
    Genera un reporte comparativo de todos los modelos
    """
    print("\n" + "="*80)
    print(" " * 25 + "📊 REPORTE DE EVALUACIÓN")
    print("="*80 + "\n")
    
    # Calcular promedios por modelo
    model_averages = {}
    
    for model_name, model_results in results.items():
        avg_scores = {
            'overall': 0,
            'keyword_coverage': 0,
            'includes_code': 0,
            'well_structured': 0,
            'category_specific': 0
        }
        
        for result in model_results:
            for metric, score in result['scores'].items():
                avg_scores[metric] += score
        
        # Calcular promedios
        num_tests = len(model_results)
        for metric in avg_scores:
            avg_scores[metric] /= num_tests
        
        model_averages[model_name] = avg_scores
    
    # Mostrar tabla comparativa
    print(f"{'Modelo':<25} | {'Overall':<10} | {'Keywords':<10} | {'Code':<10} | {'Structure':<10} | {'Category':<10}")
    print("-" * 80)
    
    for model_name, avg_scores in model_averages.items():
        print(
            f"{model_name:<25} | "
            f"{avg_scores['overall']:>8.1%} | "
            f"{avg_scores['keyword_coverage']:>8.1%} | "
            f"{avg_scores['includes_code']:>8.1%} | "
            f"{avg_scores['well_structured']:>8.1%} | "
            f"{avg_scores['category_specific']:>8.1%}"
        )
    
    # Encontrar el mejor modelo
    best_model = max(model_averages.items(), key=lambda x: x[1]['overall'])
    print("\n" + "="*80)
    print(f"🏆 Mejor modelo: {best_model[0]} con score overall de {best_model[1]['overall']:.1%}")
    print("="*80)
    
    # Guardar resultados detallados
    with open('evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n✅ Resultados detallados guardados en: evaluation_results.json")

# Ejecutar evaluación
if __name__ == "__main__":
    print("🚀 Iniciando evaluación del agente DataStage...\n")
    results = run_evaluation()
    generate_report(results)
```

### Métricas Recomendadas

Para el agente DataStage, estas son las métricas clave:

```python
# metrics.py
EVALUATION_METRICS = {
    "keyword_coverage": {
        "description": "% de palabras clave técnicas esperadas presentes",
        "weight": 0.25,
        "threshold": 0.70  # Mínimo 70% de keywords
    },
    "includes_code": {
        "description": "Contiene código PySpark ejecutable",
        "weight": 0.30,  # Muy importante para migración
        "threshold": 1.0
    },
    "well_structured": {
        "description": "Respuesta bien organizada con headers y explicaciones",
        "weight": 0.20,
        "threshold": 0.75
    },
    "category_specific": {
        "description": "Cumple requisitos específicos de la categoría",
        "weight": 0.25,
        "threshold": 0.80
    }
}
```

---

## Opción 2: GitHub Models

### Ventajas
- ✅ Acceso rápido desde GitHub Copilot
- ✅ No requiere configuración de Azure
- ✅ Ideal para prototipado

### Limitaciones
- ❌ Menos modelos disponibles
- ❌ Sin métricas integradas
- ❌ Sin logging automático
- ❌ Rate limits más restrictivos

### Uso Básico

```python
# evaluate_with_github_models.py
import requests
import os

# Lista de modelos en GitHub Models
GITHUB_MODELS = [
    "gpt-4o",
    "claude-3.5-sonnet",
    "phi-3-medium",
    "llama-3-70b"
]

def call_github_model(model_name, query):
    """
    Llama a un modelo de GitHub Models
    """
    url = f"https://models.github.com/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "You are @datastageagent, expert in DataStage to Databricks migration."
            },
            {
                "role": "user",
                "content": query
            }
        ],
        "temperature": 0.7
    }
    
    response = requests.post(url, json=payload, headers=headers)
    return response.json()['choices'][0]['message']['content']

# Ejecutar pruebas simples
for model in GITHUB_MODELS:
    print(f"\nTesting {model}...")
    response = call_github_model(
        model, 
        "Traduce: Trim(Upcase(FirstName))"
    )
    print(f"Response: {response[:200]}...")
```

---

## Opción 3: Local con AI Toolkit

### Ventajas
- ✅ Control total sobre la evaluación
- ✅ Sin costos de API
- ✅ Privacidad de datos

### Limitaciones
- ❌ Requiere más configuración
- ❌ Modelos locales tienen menos capacidad
- ❌ Necesitas hardware potente

### Script con AI Toolkit

```python
# evaluate_local.py
"""
Evaluación local usando AI Toolkit de VS Code
"""
from promptflow import PFClient
from promptflow.entities import Run

# Crear cliente
pf = PFClient()

# Configurar flujo de evaluación
evaluation_flow = {
    "name": "datastage-agent-eval",
    "description": "Evalúa agente DataStage con diferentes modelos",
    "inputs": {
        "query": {"type": "string"},
        "expected_keywords": {"type": "list"}
    },
    "outputs": {
        "response": {"type": "string"},
        "score": {"type": "number"}
    }
}

# Ejecutar evaluación
run = pf.run(
    flow="./evaluation_flow",
    data="./test_queries.jsonl",
    column_mapping={
        "query": "${data.query}",
        "expected_keywords": "${data.expected_keywords}"
    }
)

# Ver resultados
pf.get_details(run)
```

---

## Dataset de Evaluación Completo

Crea un archivo `test_dataset.json` con casos de prueba comprehensivos:

```json
{
  "test_cases": [
    {
      "id": "expr_001",
      "category": "expression_translation",
      "query": "Traduce: Trim(Upcase(FirstName)) : \" \" : Trim(Upcase(LastName))",
      "expected_output": {
        "keywords": ["F.concat", "F.trim", "F.upper"],
        "must_have_code": true,
        "min_length": 50
      }
    },
    {
      "id": "expr_002",
      "category": "expression_translation",
      "query": "Traduce: If IsNull(Status) Then \"UNKNOWN\" Else Upcase(Trim(Status))",
      "expected_output": {
        "keywords": ["F.when", "isNull", "otherwise", "F.upper"],
        "must_have_code": true
      }
    },
    {
      "id": "full_001",
      "category": "full_migration",
      "query": "Migra test-artifacts/01_simple_customer_etl.dsx a Databricks",
      "expected_output": {
        "keywords": ["spark.read", "withColumn", "Delta Lake", "write.format"],
        "must_have_code": true,
        "min_length": 500
      }
    },
    {
      "id": "comp_001",
      "category": "component_explanation",
      "query": "Explica cómo migrar un Aggregator stage",
      "expected_output": {
        "keywords": ["groupBy", "agg", "sum", "count"],
        "must_have_code": true
      }
    },
    {
      "id": "pattern_001",
      "category": "pattern_explanation",
      "query": "Cómo implemento SCD Type 2?",
      "expected_output": {
        "keywords": ["MERGE", "effective_date", "is_current"],
        "must_have_code": true,
        "min_length": 300
      }
    }
  ]
}
```

---

## Recomendación Final

### Para tu caso (Agente DataStage):

**🏆 Opción 1: Azure AI Foundry** es la mejor opción porque:

1. **Múltiples modelos**: Puedes probar GPT-4, Claude Sonnet, Mistral en un solo pipeline
2. **Métricas específicas**: Puedes crear evaluadores personalizados para:
   - Exactitud de traducción de expresiones
   - Calidad del código generado
   - Completitud de la migración
3. **Tracking**: Todas las ejecuciones quedan registradas para análisis
4. **Escalabilidad**: Puedes evaluar 100+ casos de prueba en paralelo

### Plan de Acción Inmediato:

1. **Crear dataset de evaluación** (30 casos de prueba mínimo)
2. **Configurar Azure AI Foundry** (1-2 horas)
3. **Ejecutar primera evaluación** con 2-3 modelos
4. **Analizar resultados** y ajustar el agente
5. **Re-evaluar** después de cambios

¿Quieres que te ayude a implementar alguna de estas opciones?
