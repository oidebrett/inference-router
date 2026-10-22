"""Render the LiteLLM proxy config for the PII router.

    python -m pii_router.render_config /etc/pii-router/config.yaml

Reads OLLAMA_MODEL and OLLAMA_BASE_URL for the local backend. The cloud backend is
CLOUD_BASE_URL + CLOUD_MODEL (key CLOUD_API_KEY) when set, e.g. the customer's
Vast GPU; otherwise NVIDIA_BASE_URL + NVIDIA_MODEL (key NVIDIA_API_KEY), which is
what AgentGateway uses. FALLBACK_BASE_URL + FALLBACK_MODEL (key FALLBACK_API_KEY)
add the backend non-PII traffic uses while the cloud backend is down.
Secrets are NOT written into the file: LiteLLM resolves ``os.environ/NAME`` at
runtime from the systemd EnvironmentFile.

The config must live OUTSIDE the directory holding the ``pii_router`` package:
LiteLLM loads a callback that sits next to its config as a loose file, which
breaks the package's relative imports. Elsewhere it falls back to a normal
import via PYTHONPATH.
"""

from __future__ import annotations

import json
import os
import sys

PUBLIC = "pii-router"
CLOUD = "cloud"
FALLBACK = "cloud-fallback"
LOCAL = "ollama-local"
NVIDIA_BASE = "https://integrate.api.nvidia.com/v1"
NVIDIA_DEFAULT_MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"


def _cloud_params(env) -> dict:
    custom = env.get("CLOUD_BASE_URL", "").strip()
    if custom:
        model, base, key = env.get("CLOUD_MODEL", "").strip(), custom, "CLOUD_API_KEY"
        if not model:
            raise SystemExit("CLOUD_MODEL is required with CLOUD_BASE_URL")
    else:
        model = env.get("NVIDIA_MODEL", NVIDIA_DEFAULT_MODEL).strip()
        base, key = env.get("NVIDIA_BASE_URL", NVIDIA_BASE), "NVIDIA_API_KEY"
    return {"model": f"openai/{model}", "api_base": base.rstrip("/"), "api_key": f"os.environ/{key}"}


def build(env=os.environ) -> dict:
    pii_off = (env.get("PII_ROUTER_MODE") or "on").strip().lower() == "off"
    ollama_model = env.get("OLLAMA_MODEL", "").strip()
    if pii_off:
        return _build_unchecked(env)
    if not ollama_model:
        raise SystemExit("OLLAMA_MODEL is required (the local model PII requests are routed to)")
    ollama_base = env.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")

    local_params = {"model": f"openai/{ollama_model}", "api_base": f"{ollama_base}/v1", "api_key": "ollama"}
    # Thinking models (qwen3.5) on CPU can spend minutes reasoning before the first
    # answer token: 2,000 tokens / ~5 min and no answer vs 3 s with "none" (Oracle
    # arm64, 2026-10-22). Ollama's OpenAI endpoint honours reasoning_effort; it
    # ignores think:false. PII_ROUTER_LOCAL_REASONING_EFFORT= (empty) leaves it alone.
    effort = env.get("PII_ROUTER_LOCAL_REASONING_EFFORT", "none").strip()
    if effort:
        local_params["extra_body"] = {"reasoning_effort": effort}
    model_list = [
        # The public name points at LOCAL: if the hook is not loaded, nothing reaches the cloud.
        {"model_name": PUBLIC, "litellm_params": dict(local_params)},
        {"model_name": LOCAL, "litellm_params": dict(local_params)},
        {"model_name": CLOUD, "litellm_params": _cloud_params(env)},
    ]
    fallback_base = env.get("FALLBACK_BASE_URL", "").strip()
    if fallback_base:
        fallback_model = env.get("FALLBACK_MODEL", "").strip()
        if not fallback_model:
            raise SystemExit("FALLBACK_MODEL is required with FALLBACK_BASE_URL")
        model_list.append({"model_name": FALLBACK, "litellm_params": {
            "model": f"openai/{fallback_model}", "api_base": fallback_base.rstrip("/"),
            "api_key": "os.environ/FALLBACK_API_KEY"}})
    return {
        "model_list": model_list,
        "litellm_settings": {
            "callbacks": "pii_router.hook.proxy_handler_instance",
            # Agents send provider-specific params; the other backend may reject them.
            "drop_params": True,
            "num_retries": 0,  # a retry must not silently move a request to the other backend
        },
        "router_settings": {"enable_pre_call_checks": False},
        "general_settings": {"master_key": "os.environ/LITELLM_MASTER_KEY"},
    }


def _settings() -> dict:
    return {
        "litellm_settings": {
            "callbacks": "pii_router.hook.proxy_handler_instance",
            # Agents send provider-specific params; the other backend may reject them.
            "drop_params": True,
            "num_retries": 0,  # a retry must not silently move a request to the other backend
        },
        "router_settings": {"enable_pre_call_checks": False},
        "general_settings": {"master_key": "os.environ/LITELLM_MASTER_KEY"},
    }


def _build_unchecked(env) -> dict:
    """PII_ROUTER_MODE=off: no local model. The public name points at the cloud
    backend (there is nothing local to fail closed to); the hook still swaps in the
    fallback while the cloud is down."""
    cloud = _cloud_params(env)
    model_list = [
        {"model_name": PUBLIC, "litellm_params": dict(cloud)},
        {"model_name": CLOUD, "litellm_params": dict(cloud)},
    ]
    fallback_base = env.get("FALLBACK_BASE_URL", "").strip()
    if fallback_base:
        fallback_model = env.get("FALLBACK_MODEL", "").strip()
        if not fallback_model:
            raise SystemExit("FALLBACK_MODEL is required with FALLBACK_BASE_URL")
        model_list.append({"model_name": FALLBACK, "litellm_params": {
            "model": f"openai/{fallback_model}", "api_base": fallback_base.rstrip("/"),
            "api_key": "os.environ/FALLBACK_API_KEY"}})
    return {"model_list": model_list, **_settings()}


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    # JSON is valid YAML; avoids a PyYAML dependency at render time.
    with open(argv[0], "w", encoding="utf-8") as fh:
        json.dump(build(), fh, indent=2)
        fh.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
