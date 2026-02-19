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
import ast
import xml.etree.ElementTree as ET

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
# Azure AI Foundry — modelo → nombre del deployment que configuraste
# Ajusta estos nombres según lo que pusiste al hacer el deploy en ai.azure.com
# ---------------------------------------------------------------------------
AZURE_DEPLOYMENTS = {
    "claude-sonnet-4.6": "claude-sonnet-46",
    "claude-opus-4.6":   "claude-opus-46",
    "gpt-5":             "gpt-5",
    "gpt-5.3-codex":     "gpt-5.3-codex",
    "gpt-5.2-codex":     "gpt-5.2-codex",
    "gpt-5.1-codex":     "gpt-5.1-codex",
    "gpt-4o":            "gpt-4o",
    "gpt-4.1":           "gpt-4.1",
}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def load_agent_system_prompt(agent_file: str = AGENT_FILE, include_knowledge: bool = True) -> str:
    """
    Lee el archivo .agent.md y extrae el system prompt (contenido después del frontmatter YAML).
    El agente ES este system prompt; al inyectarlo en cualquier modelo lo replicamos.
    
    Args:
        agent_file: Ruta al archivo del agente
        include_knowledge: Si True, incluye archivos de knowledge base como contexto adicional
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

    system_prompt = "\n".join(prompt_lines).strip()
    
    # Agregar knowledge base como contexto adicional
    if include_knowledge:
        import os
        from pathlib import Path
        
        knowledge_dir = Path("knowledge")
        if knowledge_dir.exists():
            system_prompt += "\n\n## KNOWLEDGE BASE - Reference Documentation\n\n"
            system_prompt += "You have access to the following reference materials:\n\n"
            
            knowledge_files = [
                ("migration-patterns.md", "Migration patterns with complete code examples"),
                ("datastage-components.md", "Complete catalog of DataStage components and equivalents"),
                ("databricks-best-practices.md", "Databricks and Delta Lake best practices"),
                ("quick-migration-guide.md", "Quick reference guide for common migrations")
            ]
            
            for filename, description in knowledge_files:
                filepath = knowledge_dir / filename
                if filepath.exists():
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            content = f.read()
                        system_prompt += f"### {filename}\n{description}\n\n```\n{content}\n```\n\n"
                    except Exception as e:
                        print(f"⚠️  Warning: Could not load {filename}: {e}")
    
    return system_prompt


def load_test_dataset(dataset_path: str = "test_dataset.json") -> List[Dict]:
    """Carga los casos de prueba del JSON y enriquece con contenido de archivos DSX."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    test_cases = data["test_cases"]
    
    # Para casos de migración completa, cargar el contenido del archivo DSX
    for tc in test_cases:
        if tc.get("category") == "full_migration":
            # Buscar si la query menciona un archivo DSX
            query = tc.get("query", "")
            if ".dsx" in query:
                # Extraer nombre del archivo
                import re
                match = re.search(r'test-artifacts/(\S+\.dsx)', query)
                if match:
                    dsx_file = match.group(0)
                    try:
                        with open(dsx_file, "r", encoding="utf-8") as f:
                            dsx_content = f.read()
                        # Agregar el contenido DSX al query para que el modelo lo analice
                        tc["query"] = f"{query}\n\n<DSX_FILE_CONTENT>\n{dsx_content}\n</DSX_FILE_CONTENT>"
                        tc["has_dsx_content"] = True
                        print(f"  📄 Cargado contenido DSX para {tc['id']}: {dsx_file}")
                    except Exception as e:
                        print(f"  ⚠️  No se pudo cargar {dsx_file}: {e}")
                        tc["has_dsx_content"] = False
    
    return test_cases


# ---------------------------------------------------------------------------
# GitHub Models API
# ---------------------------------------------------------------------------

def call_github_models(model_id: str, system_prompt: str, user_query: str,
                       token: str, timeout: int = 60) -> Dict:
    """Llama a GitHub Models API."""
    return _call_openai_compatible(
        base_url=GITHUB_MODELS_ENDPOINT,
        api_key=token,
        model_id=model_id,
        system_prompt=system_prompt,
        user_query=user_query,
    )


def call_azure_foundry(model_id: str, system_prompt: str, user_query: str,
                       endpoint: str, api_key: str) -> Dict:
    """
    Llama a Azure AI Foundry.
    model_id debe coincidir con el nombre del deployment que creaste en ai.azure.com.
    """
    deployment = AZURE_DEPLOYMENTS.get(model_id, model_id)
    return _call_openai_compatible(
        base_url=endpoint,
        api_key=api_key,
        model_id=deployment,
        system_prompt=system_prompt,
        user_query=user_query,
    )


def _call_openai_compatible(base_url: str, api_key: str, model_id: str,
                             system_prompt: str, user_query: str) -> Dict:
    """Llamada genérica compatible con OpenAI SDK (GitHub Models y Azure AI Foundry usan el mismo formato)."""
    try:
        from openai import OpenAI

        client = OpenAI(base_url=base_url, api_key=api_key)

        t0 = time.time()
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_query},
            ],
            temperature=0.3,
            max_tokens=4096,
        )
        elapsed = round(time.time() - t0, 2)

        text  = response.choices[0].message.content or ""
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
        return {"success": False, "text": "", "elapsed_sec": 0,
                "input_tokens": 0, "output_tokens": 0,
                "error": "Instala openai: pip install openai"}
    except Exception as exc:
        return {"success": False, "text": "", "elapsed_sec": 0,
                "input_tokens": 0, "output_tokens": 0, "error": str(exc)}


def call_azure_foundry_entra(model_id: str, system_prompt: str, user_query: str,
                              endpoint: str) -> Dict:
    """
    Llama a Azure AI Foundry usando autenticación Azure AD (Entra ID).
    No requiere API key — usa el login activo de 'az login'.
    Soporta modelos de chat (gpt-4o, gpt-4.1) y de completions (gpt-5.2-codex).
    """
    try:
        from azure.identity import AzureCliCredential, get_bearer_token_provider
        from openai import AzureOpenAI

        deployment = AZURE_DEPLOYMENTS.get(model_id, model_id)

        token_provider = get_bearer_token_provider(
            AzureCliCredential(),
            "https://cognitiveservices.azure.com/.default"
        )

        client = AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider,
            api_version="2025-04-01-preview",
        )

        t0 = time.time()

        # Intentar primero chat completions; codex usa completions legacy
        try:
            response = client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_query},
                ],
                temperature=0.3,
                max_tokens=4096,
            )
            elapsed = round(time.time() - t0, 2)
            text  = response.choices[0].message.content or ""
            usage = response.usage

        except Exception as chat_err:
            # Fallback 1: completions legacy (modelos codex antiguos)
            if "OperationNotSupported" in str(chat_err) or "chatCompletion" in str(chat_err) or "does not work with the specified model" in str(chat_err):
                try:
                    prompt_text = (
                        f"{system_prompt}\n\n"
                        f"### User Question\n{user_query}\n\n"
                        f"### Assistant Response\n"
                    )
                    response = client.completions.create(
                        model=deployment,
                        prompt=prompt_text,
                        temperature=0.3,
                        max_tokens=2048,
                    )
                    elapsed = round(time.time() - t0, 2)
                    text  = response.choices[0].text or ""
                    usage = response.usage
                except Exception:
                    # Fallback 2: Responses API (gpt-5.x-codex modelos nuevos)
                    response = client.responses.create(
                        model=deployment,
                        instructions=system_prompt,
                        input=user_query,
                    )
                    elapsed = round(time.time() - t0, 2)
                    text  = response.output_text or ""
                    usage = getattr(response, "usage", None)
            else:
                raise

        # Manejar diferentes estructuras de usage según el tipo de API
        input_tokens = 0
        output_tokens = 0
        if usage:
            # API Responses usa 'input_tokens' y 'output_tokens' directamente
            input_tokens = getattr(usage, 'input_tokens', getattr(usage, 'prompt_tokens', 0))
            output_tokens = getattr(usage, 'output_tokens', getattr(usage, 'completion_tokens', 0))

        return {
            "success": True,
            "text": text,
            "elapsed_sec": elapsed,
            "input_tokens":  input_tokens,
            "output_tokens": output_tokens,
            "error": None,
        }

    except ImportError:
        return {"success": False, "text": "", "elapsed_sec": 0,
                "input_tokens": 0, "output_tokens": 0,
                "error": "Instala: pip install azure-identity openai"}
    except Exception as exc:
        return {"success": False, "text": "", "elapsed_sec": 0,
                "input_tokens": 0, "output_tokens": 0, "error": str(exc)}


# ---------------------------------------------------------------------------
# Evaluador de respuestas
# ---------------------------------------------------------------------------

class AgentEvaluator:
    """Evalúa la calidad de las respuestas del agente con métricas ponderadas mejoradas."""

    WEIGHTS = {
        "keyword_coverage":  0.15,  # Reducido
        "includes_code":     0.15,  # Reducido
        "well_structured":   0.10,  # Reducido
        "category_specific": 0.60,  # Aumentado significativamente para migraciones
    }

    def extract_python_code(self, response: str) -> List[str]:
        """Extrae todos los bloques de código Python de la respuesta."""
        code_blocks = []
        # Buscar bloques con ```python
        python_blocks = re.findall(r'```python\s*\n(.*?)```', response, re.DOTALL)
        code_blocks.extend(python_blocks)
        # Buscar bloques con ``` sin especificar lenguaje
        generic_blocks = re.findall(r'```\s*\n(.*?)```', response, re.DOTALL)
        code_blocks.extend(generic_blocks)
        return [block.strip() for block in code_blocks if block.strip()]

    def validate_python_syntax(self, code: str) -> tuple[bool, str]:
        """Valida que el código Python tenga sintaxis correcta."""
        try:
            ast.parse(code)
            return True, ""
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, str(e)

    def extract_dsx_schema(self, dsx_content: str) -> Dict[str, List[str]]:
        """Extrae columnas y stages del contenido DSX."""
        schema = {"columns": set(), "stages": [], "parameters": []}
        try:
            root = ET.fromstring(dsx_content)
            # Extraer columnas
            for col in root.findall(".//Column[@Name]"):
                schema["columns"].add(col.get("Name"))
            # Extraer stages
            for stage in root.findall(".//Record[@Type='CustomStage']/Property[@Name='Name']"):
                if stage.text:
                    schema["stages"].append(stage.text)
            # Extraer parámetros
            for param in root.findall(".//Parameter/Property[@Name='Name']"):
                if param.text:
                    schema["parameters"].append(param.text)
        except Exception:
            pass  # Si falla el parsing, devolver esquema vacío
        return schema

    def analyze_pyspark_code(self, code: str) -> Dict[str, any]:
        """Analiza código PySpark para extraer columnas, transformaciones y operaciones."""
        analysis = {
            "columns": set(),
            "has_read": False,
            "has_write": False,
            "has_delta": False,
            "has_transforms": False,
            "has_validation": False,
            "has_parameters": False,
            "best_practices": []
        }
        
        # Columnas mencionadas
        col_patterns = [r"['\"]([a-zA-Z_][a-zA-Z0-9_]*)['\"]"]
        for pattern in col_patterns:
            analysis["columns"].update(re.findall(pattern, code))
        
        # Operaciones principales
        analysis["has_read"] = bool(re.search(r"spark\.read|spark\.table", code))
        analysis["has_write"] = bool(re.search(r"\.write\.", code))
        analysis["has_delta"] = "delta" in code.lower()
        analysis["has_transforms"] = bool(re.search(r"\.withColumn|\.select|\.filter", code))
        analysis["has_validation"] = bool(re.search(r"isNotNull|isNull|filter.*!=|constraint", code, re.IGNORECASE))
        analysis["has_parameters"] = "dbutils.widgets" in code
        
        # Best practices detectadas
        if "mode('overwrite')" in code or "mode('append')" in code:
            analysis["best_practices"].append("explicit_write_mode")
        if "option('header'" in code:
            analysis["best_practices"].append("csv_headers")
        if "partitionBy" in code:
            analysis["best_practices"].append("partitioning")
        if "OPTIMIZE" in code:
            analysis["best_practices"].append("optimization")
        if "# MAGIC %md" in code or "# COMMAND" in code:
            analysis["best_practices"].append("notebook_structure")
        
        return analysis

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
            # ═══════════════════════════════════════════════════════════════════
            # EVALUACIÓN AVANZADA DE MIGRACIÓN COMPLETA DSX → DATABRICKS
            # ═══════════════════════════════════════════════════════════════════
            score = 0.0
            details = {}  # Para debugging
            
            # Extraer bloques de código
            code_blocks = self.extract_python_code(response)
            all_code = "\n".join(code_blocks)
            
            # ───────────────────────────────────────────────────────────────────
            # NIVEL 1: CORRECCIÓN TÉCNICA (40%)
            # ───────────────────────────────────────────────────────────────────
            
            # 1.1 Sintaxis válida (10%)
            syntax_score = 0.0
            if code_blocks:
                valid_blocks = 0
                for block in code_blocks:
                    is_valid, error = self.validate_python_syntax(block)
                    if is_valid:
                        valid_blocks += 1
                syntax_score = (valid_blocks / len(code_blocks)) * 0.10
            score += syntax_score
            details["syntax_valid"] = syntax_score
            
            # 1.2 Schema fidelity (15%)
            # Comparar DSX columns con columnas mencionadas en PySpark
            schema_score = 0.0
            dsx_content = test_case.get("dsx_content", "")
            if dsx_content and all_code:
                dsx_schema = self.extract_dsx_schema(dsx_content)
                pyspark_analysis = self.analyze_pyspark_code(all_code)
                
                if dsx_schema["columns"]:
                    # Calcular cuántas columnas del DSX aparecen en el código PySpark
                    matched_cols = dsx_schema["columns"].intersection(pyspark_analysis["columns"])
                    schema_score = (len(matched_cols) / len(dsx_schema["columns"])) * 0.15
            else:
                # Si no hay DSX, evaluar por presencia de columnas comunes
                if all_code:
                    pyspark_analysis = self.analyze_pyspark_code(all_code)
                    if len(pyspark_analysis["columns"]) >= 5:
                        schema_score = 0.15
                    elif len(pyspark_analysis["columns"]) >= 3:
                        schema_score = 0.10
            score += schema_score
            details["schema_fidelity"] = schema_score
            
            # 1.3 Transformaciones correctas (15%)
            # Verificar que las transformaciones DataStage se convirtieron a PySpark
            transform_score = 0.0
            transform_patterns = {
                "string_ops": ["trim", "upper", "lower", "concat"],
                "date_ops": ["datediff", "date_add", "current_date", "to_date"],
                "null_handling": ["isnull", "isnotnull", "coalesce", "when"],
                "aggregations": ["groupby", "agg", "sum", "count", "avg"]
            }
            
            found_categories = 0
            for category, patterns in transform_patterns.items():
                if any(p in all_code.lower() for p in patterns):
                    found_categories += 1
            transform_score = (found_categories / len(transform_patterns)) * 0.15
            score += transform_score
            details["transformations"] = transform_score
            
            # ───────────────────────────────────────────────────────────────────
            # NIVEL 2: COMPLETITUD (30%)
            # ───────────────────────────────────────────────────────────────────
            
            # 2.1 Todos los stages migrados (15%)
            stage_score = 0.0
            if all_code:
                pyspark_analysis = self.analyze_pyspark_code(all_code)
                stages_present = sum([
                    pyspark_analysis["has_read"],
                    pyspark_analysis["has_transforms"],
                    pyspark_analysis["has_write"]
                ])
                stage_score = (stages_present / 3) * 0.15
            score += stage_score
            details["all_stages"] = stage_score
            
            # 2.2 Parámetros extraídos (10%)
            param_score = 0.0
            dsx_params = []
            if dsx_content:
                dsx_schema = self.extract_dsx_schema(dsx_content)
                dsx_params = dsx_schema["parameters"]
            
            if dsx_params and "dbutils.widgets" in all_code:
                # Verificar que los parámetros del DSX estén como widgets
                matched_params = sum(1 for p in dsx_params if p.lower() in all_code.lower())
                param_score = (matched_params / len(dsx_params)) * 0.10
            elif "dbutils.widgets" in all_code:
                # Si no hay DSX pero hay widgets, dar crédito parcial
                param_score = 0.07
            score += param_score
            details["parameters"] = param_score
            
            # 2.3 Validaciones incluidas (5%)
            validation_score = 0.0
            if all_code:
                pyspark_analysis = self.analyze_pyspark_code(all_code)
                if pyspark_analysis["has_validation"]:
                    validation_score = 0.05
            score += validation_score
            details["validations"] = validation_score
            
            # ───────────────────────────────────────────────────────────────────
            # NIVEL 3: BEST PRACTICES (20%)
            # ───────────────────────────────────────────────────────────────────
            
            # 3.1 Delta Lake usage (8%)
            delta_score = 0.0
            if all_code:
                pyspark_analysis = self.analyze_pyspark_code(all_code)
                if pyspark_analysis["has_delta"]:
                    delta_score = 0.05
                    # Bonus por OPTIMIZE o MERGE (indicadores de uso avanzado)
                    if "OPTIMIZE" in all_code or "merge" in all_code.lower():
                        delta_score = 0.08
            score += delta_score
            details["delta_lake"] = delta_score
            
            # 3.2 Partitioning strategy (6%)
            partition_score = 0.0
            if "partitionBy" in all_code:
                partition_score = 0.06
            elif "repartition" in all_code or "coalesce" in all_code:
                partition_score = 0.03
            score += partition_score
            details["partitioning"] = partition_score
            
            # 3.3 Error handling (6%)
            error_handling_score = 0.0
            error_patterns = ["try:", "except", "raise", "assert", "logger", "logging"]
            found_error_handling = sum(1 for p in error_patterns if p in all_code.lower())
            if found_error_handling >= 2:
                error_handling_score = 0.06
            elif found_error_handling == 1:
                error_handling_score = 0.03
            score += error_handling_score
            details["error_handling"] = error_handling_score
            
            # ───────────────────────────────────────────────────────────────────
            # NIVEL 4: CALIDAD (10%)
            # ───────────────────────────────────────────────────────────────────
            
            # 4.1 Documentación (5%)
            doc_score = 0.0
            doc_indicators = ["# MAGIC %md", "##", "\"\"\"", "'''"]
            found_doc = sum(1 for ind in doc_indicators if ind in response)
            if found_doc >= 3:
                doc_score = 0.05
            elif found_doc >= 2:
                doc_score = 0.03
            elif found_doc >= 1:
                doc_score = 0.02
            score += doc_score
            details["documentation"] = doc_score
            
            # 4.2 Estructura notebook (5%)
            structure_score = 0.0
            if "# MAGIC %md" in response or "# COMMAND" in response:
                structure_score = 0.03
            if "# Databricks notebook source" in response:
                structure_score = 0.05
            score += structure_score
            details["notebook_structure"] = structure_score
            
            # ───────────────────────────────────────────────────────────────────
            # RESULTADO FINAL
            # ───────────────────────────────────────────────────────────────────
            scores["category_specific"] = min(score, 1.0)
            scores["migration_details"] = details  # Para análisis detallado

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
                   mode: str, token: str = "",
                   azure_endpoint: str = "", azure_key: str = "",
                   include_knowledge: bool = True) -> Dict:

    evaluator    = AgentEvaluator()
    system_prompt = load_agent_system_prompt(include_knowledge=include_knowledge)
    all_results   = {}

    print(f"\n{'='*80}")
    print(f"  🚀  EVALUACIÓN DEL AGENTE @datastageagent")
    print(f"{'='*80}")
    print(f"  Modo      : {mode}")
    print(f"  Modelos   : {', '.join(models)}")
    print(f"  Casos     : {len(test_cases)}")
    print(f"  System prompt: {len(system_prompt):,} caracteres")
    print(f"  Knowledge base: {'✅ Incluida' if include_knowledge else '❌ No incluida'}")
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
            elif mode == "github":
                result = call_github_models(model, system_prompt, tc["query"], token)
                resp_text = result["text"] if result["success"] else ""
                meta      = result
                if not result["success"]:
                    print(f"❌  ERROR: {result['error']}")
                    continue
            elif mode == "azure":
                result = call_azure_foundry(model, system_prompt, tc["query"], azure_endpoint, azure_key)
                resp_text = result["text"] if result["success"] else ""
                meta      = result
                if not result["success"]:
                    print(f"❌  ERROR: {result['error']}")
                    continue
            elif mode == "azure-entra":
                result = call_azure_foundry_entra(model, system_prompt, tc["query"], azure_endpoint)
                resp_text = result["text"] if result["success"] else ""
                meta      = result
                if not result["success"]:
                    print(f"❌  ERROR: {result['error']}")
                    continue
            else:
                raise ValueError(f"Invalid mode: {mode}")

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

    # Desglose detallado para full_migration
    print(f"\n{'='*80}")
    print(f"  🔍  DESGLOSE DETALLADO: MIGRACIONES COMPLETAS (full_migration)")
    print(f"{'='*80}")
    
    for model, results in all_results.items():
        migration_cases = [r for r in results if r["category"] == "full_migration" and "migration_details" in r["scores"]]
        if migration_cases:
            print(f"\n  📊  Modelo: {model}")
            print(f"  {'-'*76}")
            
            # Calcular promedios de cada métrica
            metrics = {}
            for case in migration_cases:
                details = case["scores"]["migration_details"]
                for metric, value in details.items():
                    metrics.setdefault(metric, []).append(value * 100)  # Convertir a porcentaje
            
            # Mostrar promedios por categoría
            print(f"\n    {'Métrica':<35} {'Promedio':>12} {'Peso':>10}")
            print(f"    {'-'*60}")
            
            # NIVEL 1: Corrección Técnica (40%)
            print(f"    {'NIVEL 1: CORRECCIÓN TÉCNICA':<35} {'':<12} {'40%':>10}")
            if "syntax_valid" in metrics:
                avg = sum(metrics["syntax_valid"]) / len(metrics["syntax_valid"])
                print(f"      {'└─ Sintaxis válida':<33} {avg:>11.1f}% {'(10%)':>10}")
            if "schema_fidelity" in metrics:
                avg = sum(metrics["schema_fidelity"]) / len(metrics["schema_fidelity"])
                print(f"      {'└─ Schema fidelity':<33} {avg:>11.1f}% {'(15%)':>10}")
            if "transformations" in metrics:
                avg = sum(metrics["transformations"]) / len(metrics["transformations"])
                print(f"      {'└─ Transformaciones correctas':<33} {avg:>11.1f}% {'(15%)':>10}")
            
            # NIVEL 2: Completitud (30%)
            print(f"\n    {'NIVEL 2: COMPLETITUD':<35} {'':<12} {'30%':>10}")
            if "all_stages" in metrics:
                avg = sum(metrics["all_stages"]) / len(metrics["all_stages"])
                print(f"      {'└─ Todos los stages':<33} {avg:>11.1f}% {'(15%)':>10}")
            if "parameters" in metrics:
                avg = sum(metrics["parameters"]) / len(metrics["parameters"])
                print(f"      {'└─ Parámetros extraídos':<33} {avg:>11.1f}% {'(10%)':>10}")
            if "validations" in metrics:
                avg = sum(metrics["validations"]) / len(metrics["validations"])
                print(f"      {'└─ Validaciones':<33} {avg:>11.1f}% {'(5%)':>10}")
            
            # NIVEL 3: Best Practices (20%)
            print(f"\n    {'NIVEL 3: BEST PRACTICES':<35} {'':<12} {'20%':>10}")
            if "delta_lake" in metrics:
                avg = sum(metrics["delta_lake"]) / len(metrics["delta_lake"])
                print(f"      {'└─ Delta Lake':<33} {avg:>11.1f}% {'(8%)':>10}")
            if "partitioning" in metrics:
                avg = sum(metrics["partitioning"]) / len(metrics["partitioning"])
                print(f"      {'└─ Partitioning':<33} {avg:>11.1f}% {'(6%)':>10}")
            if "error_handling" in metrics:
                avg = sum(metrics["error_handling"]) / len(metrics["error_handling"])
                print(f"      {'└─ Error handling':<33} {avg:>11.1f}% {'(6%)':>10}")
            
            # NIVEL 4: Calidad (10%)
            print(f"\n    {'NIVEL 4: CALIDAD':<35} {'':<12} {'10%':>10}")
            if "documentation" in metrics:
                avg = sum(metrics["documentation"]) / len(metrics["documentation"])
                print(f"      {'└─ Documentación':<33} {avg:>11.1f}% {'(5%)':>10}")
            if "notebook_structure" in metrics:
                avg = sum(metrics["notebook_structure"]) / len(metrics["notebook_structure"])
                print(f"      {'└─ Estructura notebook':<33} {avg:>11.1f}% {'(5%)':>10}")
            
            # Overall
            overall_migration = sum(r["scores"]["overall"] for r in migration_cases) / len(migration_cases)
            print(f"\n    {'-'*60}")
            print(f"    {'OVERALL (full_migration)':<35} {overall_migration*100:>11.1f}% {'100%':>10}")

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
        choices=["local", "github", "azure", "azure-entra"],
        default="local",
        help="local = simulación | github = GitHub Models API | azure = Azure AI Foundry (key) | azure-entra = Azure AI Foundry (Entra ID, sin key)"
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
    parser.add_argument("--no-knowledge", action="store_true",
                        help="No incluir knowledge base en el system prompt (solo agent.md)")

    args = parser.parse_args()

    token        = os.environ.get("GITHUB_TOKEN", "")
    azure_endpoint = os.environ.get("AZURE_ENDPOINT", "")
    azure_key      = os.environ.get("AZURE_API_KEY", "")

    if args.mode == "github" and not token:
        print("❌  Falta GITHUB_TOKEN. Ejecuta:")
        print("    $env:GITHUB_TOKEN = 'ghp_xxxxxxxxxxxxxxxxxxxx'")
        return

    if args.mode == "azure" and not (azure_endpoint and azure_key):
        print("❌  Faltan credenciales de Azure AI Foundry. Ejecuta:")
        print("    $env:AZURE_ENDPOINT = 'https://TU-PROYECTO.openai.azure.com'")
        print("    $env:AZURE_API_KEY  = 'tu_api_key'")
        return

    if args.mode == "azure-entra" and not azure_endpoint:
        print("❌  Falta el endpoint de Azure AI Foundry. Ejecuta:")
        print("    $env:AZURE_ENDPOINT = 'https://admin-mltpgh9c-eastus2.cognitiveservices.azure.com/'")
        print("    (No necesitas API key — se usa tu login de 'az login')")
        return

    test_cases = load_test_dataset(args.dataset)
    if args.limit:
        test_cases = test_cases[: args.limit]
        print(f"⚠️  Limitado a {args.limit} casos de prueba")

    results = run_evaluation(test_cases, args.models, args.mode,
                             token=token,
                             azure_endpoint=azure_endpoint,
                             azure_key=azure_key,
                             include_knowledge=not args.no_knowledge)
    print_report(results, args.output)


if __name__ == "__main__":
    main()
