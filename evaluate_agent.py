"""
Evaluador del Agente @datastageagent con GitHub Models API
==========================================================

Cómo funciona:
  Un agente de GitHub Copilot (.agent.md) es un system prompt + modelo.
  Este script inyecta el system prompt del agente en GitHub Models API
  y compara el rendimiento entre diferentes modelos disponibles.

Uso:
  # 1. Configurar token de GitHub
  $env:GITHUB_TOKEN = "ghp_xxxxxxxxxxxxxxxxxxxx"

  # 2. Ejecutar evaluación (modo local sin API, para ver cómo funciona)
  python evaluate_agent.py --mode local

  # 3. Evaluar contra GitHub Models API real
  python evaluate_agent.py --mode github --models gpt-5.3-codex claude-sonnet-4.6

  # 4. Limitar casos para prueba rápida
  python evaluate_agent.py --mode github --limit 5

Requerimientos:
  pip install openai requests
"""

import json
import argparse
import os
import time
from datetime import datetime
from typing import List, Dict
import re

# ---------------------------------------------------------------------------
# Modelos disponibles en GitHub Copilot (febrero 2026)
# Ajustar según lo que veas en el selector de modelos
# ---------------------------------------------------------------------------
AVAILABLE_MODELS = {
    "gpt-5.3-codex":           "gpt-5.3-codex",           # 272K/128K – código especializado
    "gpt-5.2-codex":           "gpt-5.2-codex",           # 272K/128K – alternativa estable
    "gpt-5.1-codex-max":       "gpt-5.1-codex-max",       # 128K/128K – output grande
    "claude-sonnet-4.6":       "claude-sonnet-4.6",       # 128K/32K  – balance
    "claude-opus-4.6":         "claude-opus-4.6",         # 128K/64K  – máxima calidad
    "gemini-2.5-pro":          "gemini-2.5-pro",          # 109K/64K  – alternativa Google
    "gpt-5.1":                 "gpt-5.1",                 # 128K/64K  – GPT-5 estándar
}

# Modelos recomendados para @datastageagent
DEFAULT_MODELS = ["gpt-5.3-codex", "claude-sonnet-4.6", "claude-opus-4.6"]

GITHUB_MODELS_ENDPOINT = "https://models.inference.ai.azure.com"
AGENT_FILE = ".github/agents/datastageagent.agent.md"


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def load_agent_system_prompt(agent_file: str = AGENT_FILE) -> str:
    """
    Lee el archivo .agent.md y extrae el system prompt (contenido después del frontmatter YAML).
    El agente ES este system prompt; al inyectarlo en cualquier modelo lo replicamos.
    """
    with open(agent_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Saltar bloque YAML (entre --- y ---)
    lines = content.splitlines()
    in_frontmatter = False
    prompt_lines = []
    yaml_count = 0

    for line in lines:
        if line.strip() == "---":
            yaml_count += 1
            in_frontmatter = yaml_count < 2
            continue
        if not in_frontmatter:
            prompt_lines.append(line)

    return "\n".join(prompt_lines).strip()


def load_test_dataset(dataset_path: str = "test_dataset.json") -> List[Dict]:
    """Carga los casos de prueba del JSON."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["test_cases"]


# ---------------------------------------------------------------------------
# GitHub Models API
# ---------------------------------------------------------------------------

def call_github_models(model_id: str, system_prompt: str, user_query: str,
                       token: str, timeout: int = 60) -> Dict:
    """
    Llama a GitHub Models API con el system prompt del agente y una pregunta.
    Retorna el texto de la respuesta y metadata de uso.
    """
    try:
        from openai import OpenAI

        client = OpenAI(
            base_url=GITHUB_MODELS_ENDPOINT,
            api_key=token,
        )

        t0 = time.time()
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_query},
            ],
            temperature=0.3,   # Bajo para respuestas reproducibles
            max_tokens=4096,
        )
        elapsed = round(time.time() - t0, 2)

        text = response.choices[0].message.content or ""
        usage = response.usage

        return {
            "success": True,
            "text": text,
            "elapsed_sec": elapsed,
            "input_tokens":  usage.prompt_tokens     if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
            "error": None,
        }

    except ImportError:
        return {
            "success": False, "text": "",
            "elapsed_sec": 0, "input_tokens": 0, "output_tokens": 0,
            "error": "Instala openai: pip install openai",
        }
    except Exception as exc:
        return {
            "success": False, "text": "",
            "elapsed_sec": 0, "input_tokens": 0, "output_tokens": 0,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Evaluador de respuestas
# ---------------------------------------------------------------------------

class AgentEvaluator:
    """Evalúa la calidad de las respuestas del agente con métricas ponderadas."""

    WEIGHTS = {
        "keyword_coverage":  0.25,
        "includes_code":     0.30,
        "well_structured":   0.20,
        "category_specific": 0.25,
    }

    def evaluate(self, test_case: Dict, response: str) -> Dict[str, float]:
        scores = {}

        # 1. Keyword Coverage
        keywords = test_case.get("expected_keywords", [])
        if keywords:
            found = sum(1 for kw in keywords if kw.lower() in response.lower())
            scores["keyword_coverage"] = found / len(keywords)
        else:
            scores["keyword_coverage"] = 1.0

        # 2. Includes Code
        code_signals = ["```python", "```", "F.", "spark.", "df.", ".withColumn", ".filter", "import "]
        has_code = any(s in response for s in code_signals)
        if test_case.get("must_have_code", False):
            scores["includes_code"] = 1.0 if has_code else 0.0
        else:
            scores["includes_code"] = 1.0

        # 3. Well Structured
        has_headers  = any(m in response for m in ["##", "###", "**"])
        has_bullets  = any(m in response for m in ["- ", "* ", "1. "])
        min_len      = test_case.get("min_length", 0)
        long_enough  = len(response) >= min_len
        structure = 0.0
        if has_headers:  structure += 0.4
        if has_bullets:  structure += 0.3
        if long_enough:  structure += 0.3
        scores["well_structured"] = min(structure, 1.0)

        # 4. Category-specific
        cat = test_case.get("category", "")
        if cat == "expression_translation":
            ds  = any(w in response for w in ["DataStage", "BASIC", "expression"])
            ps  = any(w in response for w in ["PySpark", "Spark", "F."])
            scores["category_specific"] = 1.0 if (ds and ps) else 0.5

        elif cat == "full_migration":
            stages = ["read", "transform", "write", "validate", "parameter", "widget"]
            found  = sum(1 for s in stages if s.lower() in response.lower())
            scores["category_specific"] = min(found / 4, 1.0)

        elif cat == "component_explanation":
            purpose = any(w in response.lower() for w in ["propósito", "purpose", "qué hace", "what"])
            equiv   = any(w in response.lower() for w in ["pyspark", "spark", "equivalent", "equivalente"])
            example = "```" in response or "df." in response
            s = (0.4 if purpose else 0) + (0.3 if equiv else 0) + (0.3 if example else 0)
            scores["category_specific"] = s

        elif cat == "pattern_explanation":
            has_block  = "```" in response
            has_steps  = any(w in response.lower() for w in ["paso", "step", "primero", "first"])
            detailed   = len(response) >= test_case.get("min_length", 300)
            s = (0.5 if has_block else 0) + (0.25 if has_steps else 0) + (0.25 if detailed else 0)
            scores["category_specific"] = s

        elif cat == "optimization":
            opt_kws = ["OPTIMIZE", "Z-ORDER", "partition", "cache", "broadcast", "AQE"]
            found   = sum(1 for kw in opt_kws if kw in response)
            scores["category_specific"] = min(found / 3, 1.0)

        else:
            has_list = any(m in response for m in ["- ", "* ", "1."])
            detailed = len(response) >= 200
            scores["category_specific"] = 1.0 if (has_list and detailed) else 0.5

        # Weighted overall
        scores["overall"] = sum(scores[k] * self.WEIGHTS[k] for k in self.WEIGHTS)
        return scores


# ---------------------------------------------------------------------------
# Simulación local (sin API)
# ---------------------------------------------------------------------------

LOCAL_RESPONSES = {
    "expression_translation": (
        "## DataStage Expression\n`Trim(Upcase(col))`\n\n"
        "## PySpark Equivalent\n```python\nF.trim(F.upper(F.col('col')))\n```\n\n"
        "**Explanation**: DataStage `Upcase` → PySpark `F.upper`, `Trim` → `F.trim`.\n"
        "- DataStage concatenation `:` → PySpark `F.concat`\n"
        "- PySpark uses functional style with `F.col()` references"
    ),
    "full_migration": (
        "# Migrated Notebook\n\n## Parameters\n```python\ndbutils.widgets.text('INPUT', '/path')\n```\n\n"
        "## Read\n```python\ndf = spark.read.format('csv').option('header','true').load(INPUT_PATH)\n```\n\n"
        "## Transform\n```python\ndf_t = df.withColumn('col', F.trim(F.upper(F.col('col'))))\n```\n\n"
        "## Validate\n```python\ndf_v = df_t.filter(F.col('id').isNotNull())\n```\n\n"
        "## Write (Delta Lake)\n```python\ndf_v.write.format('delta').mode('overwrite').save(OUTPUT_PATH)\n```"
    ),
    "component_explanation": (
        "## Aggregator Stage\n\n**Purpose**: Groups data and performs aggregations.\n\n"
        "## PySpark Equivalent\n```python\ndf_agg = df.groupBy('key').agg(\n"
        "    F.sum('amount').alias('total'),\n    F.count('*').alias('cnt')\n)\n```\n\n"
        "- **Equivalent**: `groupBy().agg()` in PySpark\n- Use `F.broadcast()` for small dimension tables"
    ),
    "pattern_explanation": (
        "## SCD Type 2 in Databricks\n\n**Step 1**: Prepare source data\n```python\ndf_src = df.withColumn('eff_date', F.current_date())\n```\n\n"
        "**Step 2**: MERGE with Delta\n```python\nfrom delta.tables import DeltaTable\n"
        "DeltaTable.forPath(spark, path).alias('t').merge(\n    df_src.alias('s'), 't.key = s.key AND t.is_current=true'\n"
        ").whenMatchedUpdate(set={'is_current':'false','end_date':'current_date()'})\n"
        ".whenNotMatchedInsert(values={...}).execute()\n```"
    ),
}

def simulate_response(test_case: Dict, _model: str) -> str:
    return LOCAL_RESPONSES.get(
        test_case.get("category", ""),
        f"Simulated response for: {test_case['query']}"
    )


# ---------------------------------------------------------------------------
# Ejecución y reporte
# ---------------------------------------------------------------------------

def run_evaluation(test_cases: List[Dict], models: List[str],
                   mode: str, token: str = "") -> Dict:

    evaluator    = AgentEvaluator()
    system_prompt = load_agent_system_prompt()
    all_results   = {}

    print(f"\n{'='*80}")
    print(f"  🚀  EVALUACIÓN DEL AGENTE @datastageagent")
    print(f"{'='*80}")
    print(f"  Modo      : {mode}")
    print(f"  Modelos   : {', '.join(models)}")
    print(f"  Casos     : {len(test_cases)}")
    print(f"  System prompt: {len(system_prompt)} caracteres")
    print(f"{'='*80}\n")

    for model in models:
        print(f"\n📊  Modelo: {model}")
        print(f"  {'-'*70}")
        model_results = []

        for i, tc in enumerate(test_cases, 1):
            print(f"  [{i:02}/{len(test_cases):02}] {tc['id']} ({tc['category']})...", end=" ", flush=True)

            if mode == "local":
                resp_text = simulate_response(tc, model)
                meta = {"elapsed_sec": 0, "input_tokens": 0, "output_tokens": 0, "error": None}
            else:
                result = call_github_models(model, system_prompt, tc["query"], token)
                resp_text = result["text"] if result["success"] else ""
                meta      = result
                if not result["success"]:
                    print(f"❌  ERROR: {result['error']}")
                    continue

            scores = evaluator.evaluate(tc, resp_text)
            pct    = scores["overall"] * 100
            icon   = "✅" if pct >= 70 else ("⚠️ " if pct >= 50 else "❌")
            print(f"{icon} {pct:.1f}%  ({meta['elapsed_sec']}s)")

            model_results.append({
                "test_id": tc["id"],
                "category": tc["category"],
                "difficulty": tc.get("difficulty", "medium"),
                "query": tc["query"],
                "response_length": len(resp_text),
                "elapsed_sec": meta["elapsed_sec"],
                "output_tokens": meta["output_tokens"],
                "scores": scores,
            })

        all_results[model] = model_results

    return all_results


def print_report(all_results: Dict, output_file: str = "evaluation_report.json"):
    print(f"\n{'='*80}")
    print(f"  📊  REPORTE COMPARATIVO")
    print(f"{'='*80}\n")

    model_stats = {}
    for model, results in all_results.items():
        if not results:
            continue
        n = len(results)
        agg = {k: sum(r["scores"][k] for r in results) / n
               for k in AgentEvaluator.WEIGHTS}
        agg["overall"] = sum(r["scores"]["overall"] for r in results) / n
        agg["avg_time"] = sum(r["elapsed_sec"] for r in results) / n

        by_cat = {}
        for r in results:
            c = r["category"]
            by_cat.setdefault(c, []).append(r["scores"]["overall"])
        agg["by_category"] = {c: sum(v)/len(v) for c, v in by_cat.items()}
        model_stats[model] = agg

    # Tabla general
    header = f"{'Modelo':<25} | {'Overall':>8} | {'Keywords':>8} | {'Code':>8} | {'Structure':>9} | {'Category':>9} | {'Tiempo':>7}"
    print(header)
    print("-" * len(header))
    for m, s in model_stats.items():
        print(
            f"{m:<25} | {s['overall']:>7.1%} | {s['keyword_coverage']:>8.1%} | "
            f"{s['includes_code']:>8.1%} | {s['well_structured']:>9.1%} | "
            f"{s['category_specific']:>9.1%} | {s['avg_time']:>6.1f}s"
        )

    # Mejor modelo
    if model_stats:
        best = max(model_stats.items(), key=lambda x: x[1]["overall"])
        print(f"\n  🏆  MEJOR MODELO: {best[0]}  →  {best[1]['overall']:.1%} overall\n")

    # Por categoría
    all_cats = sorted({c for s in model_stats.values() for c in s["by_category"]})
    if all_cats:
        print(f"\n{'Categoría':<30}", end="")
        for m in model_stats:
            print(f"  {m[:18]:<18}", end="")
        print()
        print("-" * (30 + 20 * len(model_stats)))
        for cat in all_cats:
            print(f"{cat:<30}", end="")
            for s in model_stats.values():
                val = s["by_category"].get(cat, 0)
                print(f"  {val:>16.1%}  ", end="")
            print()

    # Guardar JSON
    report = {
        "generated_at": datetime.now().isoformat(),
        "summary": model_stats,
        "details": all_results,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  ✅  Reporte guardado en: {output_file}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evalúa el agente @datastageagent con diferentes modelos de GitHub Copilot"
    )
    parser.add_argument(
        "--mode",
        choices=["local", "github"],
        default="local",
        help="local = simulación sin API | github = llama a GitHub Models API (requiere GITHUB_TOKEN)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=f"Modelos a evaluar. Disponibles: {', '.join(AVAILABLE_MODELS)}"
    )
    parser.add_argument("--dataset", default="test_dataset.json")
    parser.add_argument("--output",  default="evaluation_report.json")
    parser.add_argument("--limit",   type=int, default=None,
                        help="Limitar nº de casos (útil para pruebas rápidas)")

    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN", "")
    if args.mode == "github" and not token:
        print("❌  Falta GITHUB_TOKEN. Ejecuta:")
        print("    $env:GITHUB_TOKEN = 'ghp_xxxxxxxxxxxxxxxxxxxx'")
        return

    test_cases = load_test_dataset(args.dataset)
    if args.limit:
        test_cases = test_cases[: args.limit]
        print(f"⚠️  Limitado a {args.limit} casos de prueba")

    results = run_evaluation(test_cases, args.models, args.mode, token)
    print_report(results, args.output)


if __name__ == "__main__":
    main()

    """Evaluador personalizado para el agente DataStage"""
    
    def __init__(self):
        self.metrics = {
            "keyword_coverage": 0.25,   # 25% del score
            "includes_code": 0.30,       # 30% del score
            "well_structured": 0.20,     # 20% del score
            "category_specific": 0.25    # 25% del score
        }
    
    def evaluate(self, test_case: Dict, response: str) -> Dict[str, float]:
        """
        Evalúa una respuesta del agente contra un caso de prueba
        
        Args:
            test_case: Caso de prueba con query, expected_keywords, etc.
            response: Respuesta generada por el modelo
            
        Returns:
            Dict con scores para cada métrica
        """
        scores = {}
        
        # 1. Keyword Coverage
        expected_keywords = test_case.get("expected_keywords", [])
        if expected_keywords:
            keywords_found = sum(
                1 for kw in expected_keywords 
                if kw.lower() in response.lower()
            )
            scores['keyword_coverage'] = keywords_found / len(expected_keywords)
        else:
            scores['keyword_coverage'] = 1.0
        
        # 2. Includes Code
        code_markers = ['```python', '```', 'F.', 'spark.', 'df.', '.withColumn', '.filter']
        has_code = any(marker in response for marker in code_markers)
        
        if test_case.get("must_have_code", False):
            scores['includes_code'] = 1.0 if has_code else 0.0
        else:
            scores['includes_code'] = 1.0  # No requerido
        
        # 3. Well Structured
        has_headers = any(marker in response for marker in ['##', '###', '**'])
        has_bullet_points = any(marker in response for marker in ['- ', '* ', '1. '])
        response_length = len(response)
        min_length = test_case.get("min_length", 0)
        
        structure_score = 0.0
        if has_headers:
            structure_score += 0.4
        if has_bullet_points:
            structure_score += 0.3
        if response_length >= min_length:
            structure_score += 0.3
        
        scores['well_structured'] = min(structure_score, 1.0)
        
        # 4. Category Specific
        category = test_case.get("category", "")
        
        if category == "expression_translation":
            # Debe mostrar DataStage y PySpark lado a lado
            has_datastage_label = any(word in response for word in ["DataStage", "BASIC", "expression"])
            has_pyspark_label = any(word in response for word in ["PySpark", "Spark", "F."])
            has_comparison = has_datastage_label and has_pyspark_label
            scores['category_specific'] = 1.0 if has_comparison else 0.5
            
        elif category == "full_migration":
            # Debe incluir múltiples secciones del pipeline
            stages = ["read", "transform", "write", "validate", "parameter", "widget"]
            stages_mentioned = sum(1 for s in stages if s.lower() in response.lower())
            scores['category_specific'] = min(stages_mentioned / 4, 1.0)
            
        elif category == "component_explanation":
            # Debe explicar propósito y mostrar equivalente
            has_purpose = any(word in response.lower() for word in 
                            ["purpose", "propósito", "qué hace", "what", "usado para"])
            has_equivalent = any(word in response.lower() for word in 
                               ["pyspark", "spark", "equivalent", "equivalente"])
            has_example = "```" in response or "df." in response
            
            explanation_score = 0
            if has_purpose:
                explanation_score += 0.4
            if has_equivalent:
                explanation_score += 0.3
            if has_example:
                explanation_score += 0.3
            
            scores['category_specific'] = explanation_score
            
        elif category == "pattern_explanation":
            # Debe incluir código completo y explicación detallada
            has_code_block = "```" in response
            has_steps = any(word in response.lower() for word in ["paso", "step", "primero", "first"])
            response_is_detailed = len(response) >= test_case.get("min_length", 300)
            
            pattern_score = 0
            if has_code_block:
                pattern_score += 0.5
            if has_steps:
                pattern_score += 0.25
            if response_is_detailed:
                pattern_score += 0.25
            
            scores['category_specific'] = pattern_score
            
        elif category == "optimization":
            # Debe mencionar técnicas específicas
            optimization_keywords = ["OPTIMIZE", "Z-ORDER", "partition", "cache", "broadcast", "AQE"]
            optimizations_mentioned = sum(1 for kw in optimization_keywords if kw in response)
            scores['category_specific'] = min(optimizations_mentioned / 3, 1.0)
            
        elif category == "troubleshooting":
            # Debe identificar problema y dar solución
            has_problem_id = any(word in response.lower() for word in ["causa", "cause", "porque", "reason"])
            has_solution = any(word in response.lower() for word in ["solución", "solution", "fix", "resolver"])
            scores['category_specific'] = 1.0 if (has_problem_id and has_solution) else 0.5
            
        else:  # best_practices
            has_list = any(marker in response for marker in ['- ', '* ', '1.', '2.'])
            has_recommendations = len(response) >= 200
            scores['category_specific'] = 1.0 if (has_list and has_recommendations) else 0.5
        
        # Calculate weighted overall score
        overall = sum(scores[metric] * self.metrics[metric] for metric in scores)
        scores['overall'] = overall
        
        return scores


def simulate_agent_response(test_case: Dict, model_name: str) -> str:
    """
    Simula respuesta del agente (para modo local sin API)
    En producción, esto llamaría a la API del modelo real
    """
    query = test_case['query']
    category = test_case['category']
    
    # Respuestas simuladas básicas por categoría
    responses = {
        "expression_translation": f"""
## DataStage Expression
`{query.split(":")[-1] if ":" in query else "DataStage expression"}`

## PySpark Equivalent
```python
# Translation:
F.concat(F.trim(F.upper(F.col("FirstName"))), F.lit(" "), F.trim(F.upper(F.col("LastName"))))
```

**Explanation**: DataStage uses `Trim()`, `Upcase()` which map to PySpark's `F.trim()` and `F.upper()`.
The colon `:` operator concatenates strings, which in PySpark is `F.concat()`.
""",
        "full_migration": f"""
# DataStage Job Migration

I'll migrate this DataStage job to Databricks with PySpark.

## Parameters Setup
```python
dbutils.widgets.text("INPUT_PATH", "/data/input")
dbutils.widgets.text("OUTPUT_PATH", "/data/output")
```

## Read Stage
```python
df = spark.read.format("csv").option("header", "true").load(INPUT_PATH)
```

## Transform Stage
```python
df_transformed = df.withColumn("cleaned_col", F.trim(F.upper(F.col("col"))))
```

## Validation
```python
df_validated = df_transformed.filter(F.col("id").isNotNull())
```

## Write Stage  
```python
df_validated.write.format("delta").mode("overwrite").save(OUTPUT_PATH)
```

This modernizes from CSV to Delta Lake for ACID compliance.
""",
        "component_explanation": f"""
## DataStage Aggregator Stage

**Purpose**: Groups data by keys and performs aggregations (sum, count, max, etc.)

## PySpark Equivalent

```python
# Aggregator stage translation
df_aggregated = df.groupBy("customer_id", "region").agg(
    F.sum("amount").alias("total_amount"),
    F.count("*").alias("record_count"),
    F.max("transaction_date").alias("last_transaction")
)
```

**Key Differences**: 
- DataStage uses visual grouping configuration
- PySpark uses `groupBy()` with `agg()` for aggregations
- PySpark is more flexible with custom aggregations
""",
        "pattern_explanation": f"""
## Slowly Changing Dimension Type 2 Pattern

Here's how to implement SCD Type 2 in Databricks:

### Step 1: Prepare incoming data
```python
df_new = df_source.withColumn("effective_date", F.current_date())
```

### Step 2: Use Delta MERGE for upsert
```python
from delta.tables import DeltaTable

DeltaTable.forPath(spark, target_path).alias("target").merge(
    df_new.alias("source"),
    "target.business_key = source.business_key AND target.is_current = true"
).whenMatchedUpdate(
    condition="source.hash_value != target.hash_value",
    set={{
        "is_current": "false",
        "end_date": "current_date()"
    }}
).whenNotMatchedInsert(
    values={{
        "business_key": "source.business_key",
        "effective_date": "current_date()",
        "is_current": "true"
    }}
).execute()
```

This maintains full history with effective dating.
"""
    }
    
    # Retornar respuesta simulada según categoría
    return responses.get(category, f"Response for {category}: {query}")


def load_test_dataset(dataset_path: str = "test_dataset.json") -> List[Dict]:
    """Carga el dataset de casos de prueba"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['test_cases']


def run_evaluation(test_cases: List[Dict], models: List[str], mode: str = "local") -> Dict:
    """
    Ejecuta la evaluación completa
    
    Args:
        test_cases: Lista de casos de prueba
        models: Lista de nombres de modelos a evaluar
        mode: "local" (simulado), "azure" (Azure AI Foundry), "github" (GitHub Models)
    
    Returns:
        Dict con resultados por modelo
    """
    evaluator = AgentEvaluator()
    results = {}
    
    print(f"\n{'='*80}")
    print(f"  🚀 INICIANDO EVALUACIÓN DEL AGENTE DATASTAGE")
    print(f"{'='*80}")
    print(f"  Modo: {mode}")
    print(f"  Modelos: {', '.join(models)}")
    print(f"  Casos de prueba: {len(test_cases)}")
    print(f"{'='*80}\n")
    
    for model_name in models:
        print(f"\n📊 Evaluando modelo: {model_name}")
        print(f"{'-'*80}")
        
        model_results = []
        
        for i, test_case in enumerate(test_cases, 1):
            test_id = test_case['id']
            category = test_case['category']
            difficulty = test_case.get('difficulty', 'medium')
            
            print(f"  [{i}/{len(test_cases)}] {test_id} ({category} - {difficulty})...", end=" ")
            
            # Obtener respuesta del modelo
            if mode == "local":
                response = simulate_agent_response(test_case, model_name)
            elif mode == "azure":
                # TODO: Implementar llamada a Azure AI Foundry
                response = f"Azure response for {test_id}"
            elif mode == "github":
                # TODO: Implementar llamada a GitHub Models
                response = f"GitHub response for {test_id}"
            else:
                raise ValueError(f"Invalid mode: {mode}")
            
            # Evaluar respuesta
            scores = evaluator.evaluate(test_case, response)
            
            result = {
                "test_id": test_id,
                "category": category,
                "difficulty": difficulty,
                "query": test_case['query'],
                "response": response,
                "scores": scores,
                "timestamp": datetime.now().isoformat()
            }
            
            model_results.append(result)
            
            # Mostrar score
            overall_pct = scores['overall'] * 100
            emoji = "✅" if overall_pct >= 70 else "⚠️" if overall_pct >= 50 else "❌"
            print(f"{emoji} {overall_pct:.1f}%")
        
        results[model_name] = model_results
    
    return results


def generate_report(results: Dict, output_file: str = "evaluation_report.json"):
    """
    Genera reporte comprehensivo de resultados
    """
    print(f"\n{'='*80}")
    print(f"  📊 REPORTE DE EVALUACIÓN")
    print(f"{'='*80}\n")
    
    # Calcular estadísticas por modelo
    model_stats = {}
    
    for model_name, model_results in results.items():
        stats = {
            'overall_avg': 0,
            'keyword_coverage_avg': 0,
            'includes_code_avg': 0,
            'well_structured_avg': 0,
            'category_specific_avg': 0,
            'by_category': {},
            'by_difficulty': {},
            'total_tests': len(model_results)
        }
        
        # Calcular promedios generales
        for result in model_results:
            scores = result['scores']
            for metric in ['overall', 'keyword_coverage', 'includes_code', 'well_structured', 'category_specific']:
                stats[f'{metric}_avg'] += scores[metric]
            
            # Por categoría
            category = result['category']
            if category not in stats['by_category']:
                stats['by_category'][category] = []
            stats['by_category'][category].append(scores['overall'])
            
            # Por dificultad
            difficulty = result['difficulty']
            if difficulty not in stats['by_difficulty']:
                stats['by_difficulty'][difficulty] = []
            stats['by_difficulty'][difficulty].append(scores['overall'])
        
        # Dividir por número de tests
        for key in ['overall_avg', 'keyword_coverage_avg', 'includes_code_avg', 
                    'well_structured_avg', 'category_specific_avg']:
            stats[key] /= stats['total_tests']
        
        # Promedios por categoría
        for category in stats['by_category']:
            scores_list = stats['by_category'][category]
            stats['by_category'][category] = sum(scores_list) / len(scores_list)
        
        # Promedios por dificultad
        for difficulty in stats['by_difficulty']:
            scores_list = stats['by_difficulty'][difficulty]
            stats['by_difficulty'][difficulty] = sum(scores_list) / len(scores_list)
        
        model_stats[model_name] = stats
    
    # Tabla comparativa de modelos
    print("## Comparación General de Modelos\n")
    print(f"{'Modelo':<25} | {'Overall':<10} | {'Keywords':<10} | {'Code':<10} | {'Structure':<10} | {'Category':<10}")
    print("-" * 95)
    
    for model_name, stats in model_stats.items():
        print(
            f"{model_name:<25} | "
            f"{stats['overall_avg']:>8.1%} | "
            f"{stats['keyword_coverage_avg']:>8.1%} | "
            f"{stats['includes_code_avg']:>8.1%} | "
            f"{stats['well_structured_avg']:>8.1%} | "
            f"{stats['category_specific_avg']:>8.1%}"
        )
    
    # Mejor modelo
    best_model = max(model_stats.items(), key=lambda x: x[1]['overall_avg'])
    print(f"\n🏆 MEJOR MODELO: {best_model[0]} con {best_model[1]['overall_avg']:.1%} overall\n")
    
    # Performance por categoría
    print("\n## Performance por Categoría\n")
    all_categories = set()
    for stats in model_stats.values():
        all_categories.update(stats['by_category'].keys())
    
    for category in sorted(all_categories):
        print(f"\n### {category}")
        for model_name, stats in model_stats.items():
            if category in stats['by_category']:
                score = stats['by_category'][category]
                print(f"  {model_name:<25} {score:>6.1%}")
    
    # Performance por dificultad
    print("\n## Performance por Dificultad\n")
    for difficulty in ['easy', 'medium', 'hard']:
        print(f"\n### {difficulty.upper()}")
        for model_name, stats in model_stats.items():
            if difficulty in stats['by_difficulty']:
                score = stats['by_difficulty'][difficulty]
                print(f"  {model_name:<25} {score:>6.1%}")
    
    # Guardar resultados completos
    report_data = {
        'evaluation_date': datetime.now().isoformat(),
        'summary': model_stats,
        'detailed_results': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"✅ Reporte completo guardado en: {output_file}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluar agente DataStage con múltiples modelos")
    parser.add_argument(
        "--mode", 
        choices=["local", "azure", "github"],
        default="local",
        help="Modo de evaluación (local=simulado, azure=Azure AI Foundry, github=GitHub Models)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-4o", "claude-sonnet-3.5", "gpt-4-turbo"],
        help="Lista de modelos a evaluar"
    )
    parser.add_argument(
        "--dataset",
        default="test_dataset.json",
        help="Ruta al archivo de dataset de prueba"
    )
    parser.add_argument(
        "--output",
        default="evaluation_report.json",
        help="Archivo de salida para el reporte"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limitar número de casos de prueba (útil para pruebas rápidas)"
    )
    
    args = parser.parse_args()
    
    # Cargar dataset
    print(f"📁 Cargando dataset: {args.dataset}")
    test_cases = load_test_dataset(args.dataset)
    
    if args.limit:
        test_cases = test_cases[:args.limit]
        print(f"⚠️  Limitado a {args.limit} casos de prueba")
    
    # Ejecutar evaluación
    results = run_evaluation(test_cases, args.models, args.mode)
    
    # Generar reporte
    generate_report(results, args.output)
    
    print("✅ Evaluación completada exitosamente!")


if __name__ == "__main__":
    main()
