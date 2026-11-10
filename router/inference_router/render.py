"""Render router.yaml into the LiteLLM proxy config.

    python -m inference_router.render /etc/inference-router/router.yaml /etc/inference-router/litellm.yaml

For every route the targets become deployments ``ir/<route>/<n>`` chained as LiteLLM
fallbacks (target 0, then 1, ...). Routes with private upstreams also get a private
chain ``ir/<route>/private/<n>`` for keep_private, and routes with a personal
subscription (the admin's Claude or ChatGPT plan) a shared chain ``ir/<route>/shared/<n>``
without it, for callers not allowed to use it. The names callers use (the route
names) point at a deployment that only returns a block message: the hook picks the
real chain, so if the hook is not loaded nothing reaches an upstream.

Secrets are not written: each upstream's key is ``os.environ/<key_env>``, resolved by
LiteLLM from the secrets env file.
"""

from __future__ import annotations

import json
import sys

from .config import Config, Route, load
from .subscriptions import CHATGPT, CLAUDE, CLAUDE_BETAS, chatgpt_signed_in

BLOCKED = "ir/blocked"
BLOCKED_TEXT = "[Inference Router] This request was blocked: the router is not active."


VARIANTS = ("all", "private", "shared")


def chain_name(route: Route, n: int, private: bool | str = False) -> str:
    variant = "private" if private is True else (private or "all")
    return f"ir/{route.name}/{'' if variant == 'all' else variant + '/'}{n}"


def _params(config: Config, target) -> dict:
    up = config.upstream(target.upstream)
    if up.kind == CLAUDE:
        # LiteLLM sends an sk-ant-oat token as Bearer with the oauth beta; the claude-code
        # beta and the identity line (hook.async_pre_call_deployment_hook) complete the shape.
        return {"model": f"anthropic/{target.model}", "api_key": f"os.environ/{up.key_env}",
                "extra_headers": {"anthropic-beta": CLAUDE_BETAS}}
    if up.kind == CHATGPT:
        return {"model": f"chatgpt/{target.model}"}   # tokens: CHATGPT_TOKEN_DIR/auth.json
    params = {"model": f"openai/{target.model}", "api_base": up.base_url, "api_key": f"os.environ/{up.key_env}"}
    effort = up.effective_reasoning_effort()
    if effort:
        params["extra_body"] = {"reasoning_effort": effort}
    return params


def blocked_params() -> dict:
    return {"model": "openai/ir-blocked", "api_base": "http://127.0.0.1:9/v1", "api_key": "none",
            "mock_response": BLOCKED_TEXT}


def _targets(config: Config, route: Route, variant: str, env=None) -> list:
    targets = config.live_targets(route, private_only=variant == "private", shared_only=variant == "shared")
    # A ChatGPT target with no saved sign-in would fail every request: leave it out until
    # the admin signs in (the sign-in restarts the router, which re-renders).
    signed_in = None
    out = []
    for t in targets:
        if config.upstream(t.upstream).kind == CHATGPT:
            if signed_in is None:
                signed_in = chatgpt_signed_in(env) if env is not None else chatgpt_signed_in()
            if not signed_in:
                continue
        out.append(t)
    return out


def chains(config: Config, env=None) -> dict[str, dict[str, list[str]]]:
    """route name -> {"all": [...], "private": [...], "shared": [...]} deployment names.
    "shared" is "all" without personal subscriptions (the same chain when there are none)."""
    out = {}
    for route in config.routes:
        names = {v: [chain_name(route, i, v) for i, _ in enumerate(_targets(config, route, v, env))]
                 for v in ("all", "private")}
        names["shared"] = ([chain_name(route, i, "shared") for i, _ in enumerate(_targets(config, route, "shared", env))]
                           if config.has_subscription(route) else names["all"])
        out[route.name] = names
    return out


def chain_targets(config: Config, route: Route, variant: str, env=None) -> list:
    """The targets behind chains()[route][variant], in order."""
    if variant == "shared" and not config.has_subscription(route):
        variant = "all"
    return _targets(config, route, variant, env)


def build(config: Config, env=None) -> dict:
    model_list = [{"model_name": name, "litellm_params": blocked_params()} for name in config.route_names()]
    model_list.append({"model_name": BLOCKED, "litellm_params": blocked_params()})
    fallbacks = []
    for route in config.routes:
        for variant in VARIANTS:
            if variant == "shared" and not config.has_subscription(route):
                continue
            targets = _targets(config, route, variant, env)
            names = [chain_name(route, i, variant) for i in range(len(targets))]
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
