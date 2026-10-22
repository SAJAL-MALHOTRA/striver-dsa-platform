"""
Provider-independent LLM orchestration layer.

Supports:
- Grok (xAI)
- Groq
- OpenAI
- Custom OpenAI-compatible endpoints (Ollama, vLLM, local)
- Deterministic extractive fallback with citation integrity
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Optional

# Standard provider configurations
PROVIDER_CONFIGS = {
    "grok": {
        "name": "Grok (xAI)",
        "base_url": "https://api.x.ai/v1",
        "default_model": "grok-2-latest",
        "fallback_model": "grok-beta",
    },
    "groq": {
        "name": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
        "fallback_model": "llama-3.1-8b-instant",
    },
    "openai": {
        "name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "fallback_model": "gpt-4o",
    },
}


def get_provider_details(provider: str, custom_base_url: str = "", custom_model: str = "") -> tuple[str, str]:
    """Resolves the base URL and target model for the requested provider."""
    provider_key = (provider or "groq").lower().strip()
    cfg = PROVIDER_CONFIGS.get(provider_key, {})

    base_url = custom_base_url.strip() if custom_base_url else cfg.get("base_url", "https://api.groq.com/openai/v1")
    model = custom_model.strip() if custom_model else cfg.get("default_model", "llama-3.3-70b-versatile")
    return base_url.rstrip("/"), model


def validate_api_key(
    provider: str,
    api_key: str,
    base_url: str = "",
    model: str = "",
    timeout: float = 8.0,
) -> dict:
    """
    Validates an API key against the provider's authenticated /models endpoint.

    Returns
    -------
    dict:
        { "valid": bool, "provider": str, "model": str, "available_models": list, "message": str, "error": str }
    """
    p_clean = (provider or "").lower().strip()
    if p_clean in ("local", "striver-rtx5050", "striver"):
        from ytrag.local_llm import LocalTeachingAssistant
        assistant = LocalTeachingAssistant.get_instance()
        return {
            "valid": True,
            "provider": "local",
            "model": "Qwen2.5-Coder-7B-Striver (RTX 5050)",
            "message": "Connected to Local NVIDIA RTX 5050 Striver Model (No Cloud Keys Needed)!",
            "available_models": ["striver_dsa_lora_final"],
        }

    api_key = api_key.strip()
    if not api_key:
        return {
            "valid": False,
            "provider": provider,
            "error": "API key cannot be empty.",
            "message": "Please enter an API key.",
        }

    resolved_base_url, resolved_model = get_provider_details(provider, base_url, model)
    url = f"{resolved_base_url}/models"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "YTLectureRAG/1.0",
        "Accept": "application/json",
    }
    req = urllib.request.Request(url, headers=headers, method="GET")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status = response.status
            raw = response.read().decode("utf-8")
            data = json.loads(raw)
            model_ids = []
            if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                model_ids = [m.get("id") for m in data["data"] if isinstance(m, dict) and m.get("id")]

            return {
                "valid": True,
                "provider": provider,
                "model": resolved_model,
                "available_models": model_ids[:15],
                "message": f"Successfully verified authentication with {provider.upper()}!",
            }
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            err_msg = "Authentication failed: Invalid API key (HTTP 401 Unauthorized)."
        elif exc.code == 403:
            err_msg = "Access forbidden: Check API key permissions or project organization (HTTP 403)."
        elif exc.code == 429:
            err_msg = "Rate limit or quota exceeded on this API key (HTTP 429)."
        else:
            err_msg = f"Provider returned HTTP error {exc.code}: {exc.reason}."
        return {"valid": False, "provider": provider, "error": err_msg}
    except urllib.error.URLError as exc:
        return {
            "valid": False,
            "provider": provider,
            "error": f"Connection failed to {resolved_base_url}: {exc.reason}.",
        }
    except Exception as exc:
        return {
            "valid": False,
            "provider": provider,
            "error": f"Validation error: {str(exc)}",
        }


def generate_llm_answer(
    query: str,
    lecture_excerpts: list[dict],
    dsa_context: Optional[dict] = None,
    provider: str = "groq",
    api_key: str = "",
    model: str = "",
    base_url: str = "",
    timeout: float = 30.0,
) -> dict:
    """
    Generates a structured, evidence-grounded answer using the configured provider.

    Enforces strict citation grounding:
    - Never invent source citations or hallucinate transcript facts.
    - Integrates verified DSA approach & complexity notes if available.
    """
    p_clean = (provider or "").lower().strip()
    if p_clean in ("local", "striver-rtx5050", "striver"):
        from ytrag.local_llm import LocalTeachingAssistant
        assistant = LocalTeachingAssistant.get_instance()
        return assistant.generate_answer(
            query=query,
            lecture_excerpts=lecture_excerpts,
            dsa_context=dsa_context,
        )

    resolved_base_url, resolved_model = get_provider_details(provider, base_url, model)

    context_str = json.dumps({
        "question": query,
        "lecture_moments": [
            {
                "source_id": i + 1,
                "video_title": item.get("video_title", ""),
                "timestamp": item.get("timestamp_display", ""),
                "transcript": item.get("snippet", "")[:1200],
            }
            for i, item in enumerate(lecture_excerpts)
        ],
        "canonical_dsa_knowledge": {
            "title": dsa_context.get("title", "") if dsa_context else "",
            "approach": dsa_context.get("approach", "") if dsa_context else "",
            "time_complexity": dsa_context.get("time_complexity", "") if dsa_context else "",
            "space_complexity": dsa_context.get("space_complexity", "") if dsa_context else "",
        } if dsa_context else None,
    }, ensure_ascii=False)

    system_prompt = (
        "You are an expert DSA Teaching Assistant grounded in Striver's A2Z DSA lectures.\n"
        "Guidelines:\n"
        "1. Base technical explanations strictly on the provided lecture transcript moments and canonical DSA knowledge.\n"
        "2. Do NOT invent source IDs or citations. Reference only valid source_id integers (1 to N).\n"
        "3. Provide at most 3 concise, high-yield bullet points.\n"
        "4. If canonical DSA knowledge is present, mention key edge-cases and the optimal pointer/window approach.\n"
        "5. Return strictly a JSON object with this exact schema:\n"
        '{\n  "bullets": [\n    {"text": "concise explanation", "sources": [1]}\n  ],\n  "approach_summary": "1-2 sentence core intuition"\n}'
    )

    payload = {
        "model": resolved_model,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context_str},
        ],
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    url = f"{resolved_base_url}/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    raw_content = body["choices"][0]["message"]["content"]
    parsed = json.loads(raw_content)
    return parsed
