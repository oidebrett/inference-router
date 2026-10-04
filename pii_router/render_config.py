"""Render the LiteLLM proxy config for the PII router.

    python -m pii_router.render_config /etc/pii-router/config.yaml

Reads OLLAMA_MODEL, OLLAMA_BASE_URL, NVIDIA_MODEL, NVIDIA_BASE_URL from the
environment. Secrets are NOT written into the file: LiteLLM resolves
``os.environ/NAME`` at runtime from the systemd EnvironmentFile.

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
CLOUD = "nvidia-cloud"
LOCAL = "ollama-local"


def build(env=os.environ) -> dict:
    ollama_model = env.get("OLLAMA_MODEL", "").strip()
    nvidia_model = env.get("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b").strip()
    if not ollama_model:
        raise SystemExit("OLLAMA_MODEL is required (the local model PII requests are routed to)")
    ollama_base = env.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    nvidia_base = env.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")

    local_params = {"model": f"openai/{ollama_model}", "api_base": f"{ollama_base}/v1", "api_key": "ollama"}
    return {
        "model_list": [
            # The public name points at LOCAL: if the hook is not loaded, nothing reaches the cloud.
            {"model_name": PUBLIC, "litellm_params": dict(local_params)},
            {"model_name": LOCAL, "litellm_params": dict(local_params)},
            {"model_name": CLOUD, "litellm_params": {
                "model": f"openai/{nvidia_model}", "api_base": nvidia_base,
                "api_key": "os.environ/NVIDIA_API_KEY"}},
        ],
        "litellm_settings": {
            "callbacks": "pii_router.hook.proxy_handler_instance",
            # Agents send provider-specific params; the other backend may reject them.
            "drop_params": True,
            "num_retries": 0,  # a retry must not silently move a request to the other backend
        },
        "router_settings": {"enable_pre_call_checks": False},
        "general_settings": {"master_key": "os.environ/LITELLM_MASTER_KEY"},
    }


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
