"""Render router.yaml into the LiteLLM proxy config.

    python -m inference_router.render /etc/inference-router/router.yaml /etc/inference-router/litellm.yaml

For every route the targets become deployments ``ir/<route>/<n>`` chained as LiteLLM
fallbacks (target 0, then 1, ...). Routes with private upstreams also get a private
chain ``ir/<route>/private/<n>`` for keep_private. The names callers use (the route
names) point at a deployment that only returns a block message: the hook picks the
real chain, so if the hook is not loaded nothing reaches an upstream.

Secrets are not written: each upstream's key is ``os.environ/<key_env>``, resolved by
LiteLLM from the secrets env file.
"""

from __future__ import annotations

import json
import sys

from .config import Config, Route, load

BLOCKED = "ir/blocked"
BLOCKED_TEXT = "[Inference Router] This request was blocked: the router is not active."


def chain_name(route: Route, n: int, private: bool = False) -> str:
    return f"ir/{route.name}/{'private/' if private else ''}{n}"


def _params(config: Config, target) -> dict:
    up = config.upstream(target.upstream)
    params = {"model": f"openai/{target.model}", "api_base": up.base_url, "api_key": f"os.environ/{up.key_env}"}
    effort = up.effective_reasoning_effort()
    if effort:
        params["extra_body"] = {"reasoning_effort": effort}
    return params


def blocked_params() -> dict:
    return {"model": "openai/ir-blocked", "api_base": "http://127.0.0.1:9/v1", "api_key": "none",
            "mock_response": BLOCKED_TEXT}


def chains(config: Config) -> dict[str, dict[str, list[str]]]:
    """route name -> {"all": [deployment names], "private": [...]} (enabled targets only)."""
    out = {}
    for route in config.routes:
        all_names = [chain_name(route, i) for i, _ in enumerate(config.live_targets(route))]
        private = [chain_name(route, i, True) for i, _ in enumerate(config.live_targets(route, private_only=True))]
        out[route.name] = {"all": all_names, "private": private}
    return out


def build(config: Config) -> dict:
    model_list = [{"model_name": name, "litellm_params": blocked_params()} for name in config.route_names()]
    model_list.append({"model_name": BLOCKED, "litellm_params": blocked_params()})
    fallbacks = []
    for route in config.routes:
        for private in (False, True):
            targets = config.live_targets(route, private_only=private)
            names = [chain_name(route, i, private) for i in range(len(targets))]
            for name, target in zip(names, targets):
                model_list.append({"model_name": name, "litellm_params": _params(config, target)})
            if len(names) > 1:
                fallbacks.append({names[0]: names[1:]})
    router_settings = {"enable_pre_call_checks": False}
    if fallbacks:
        router_settings["fallbacks"] = fallbacks
    return {
        "model_list": model_list,
        "litellm_settings": {
            "callbacks": "inference_router.hook.proxy_handler_instance",
            "drop_params": True,   # callers send provider-specific params another upstream may reject
            "num_retries": 0,      # failover is the explicit fallback chain, nothing silent
        },
        "router_settings": router_settings,
        "general_settings": {"master_key": "os.environ/LITELLM_MASTER_KEY",
                             "custom_auth": "inference_router.auth.user_api_key_auth"},
    }


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    config = load(argv[0])
    with open(argv[1], "w", encoding="utf-8") as fh:
        json.dump(build(config), fh, indent=2)   # JSON is valid YAML
        fh.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
