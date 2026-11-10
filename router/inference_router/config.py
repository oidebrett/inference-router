"""Router configuration: upstreams, routes, privacy and PII rules.

The admin owns one YAML file (default /etc/inference-router/router.yaml). It holds
no secrets: each upstream's API key lives in the secrets env file under a name
derived from the upstream id (``key_env``). The file is validated as a whole;
an invalid file is refused rather than partly applied.

    version: 1
    privacy:
      mode: mask            # off | mask | mask_check | keep_private
      laya: false           # mask_check / keep_private: also ask the Laya classifier
      rules:
        builtin: {phone: true, email: true}          # omitted types default to on
        custom:
          - {name: CLAIM_NUMBER, pattern: 'CLM-\\d{4}-\\d{7}', keywords: [claim]}
        exceptions: [support@example.com]
    upstreams:
      - {id: nvidia, name: NVIDIA, base_url: https://integrate.api.nvidia.com/v1}
      - {id: office, name: Office GPU, base_url: https://gpu.example.com/v1, private: true}
    routes:
      - name: default
        targets:
          - {upstream: nvidia, model: nvidia/nemotron-3.5-super-vl-120b-a12b}
          - {upstream: office, model: qwen3.8-27b}
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

import yaml

DEFAULT_PATH = "/etc/inference-router/router.yaml"
PRIVACY_MODES = ("off", "mask", "mask_check", "keep_private")
BUILTIN_TYPES = ("email", "phone", "iban", "payment_card", "us_ssn", "uk_nino", "ie_ppsn",
                 "passport", "date_of_birth")
VALIDATORS = ("luhn", "mod97")
# Hosted APIs that may reject reasoning_effort="none"; self-hosted vLLM gets it by default.
HOSTED_API_HOSTS = ("integrate.api.nvidia.com", "openrouter.ai", "api.openai.com", "api.anthropic.com")

_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")
_ROUTE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,63}$")
_RULE_NAME = re.compile(r"^[A-Z][A-Z0-9_]{1,31}$")


class ConfigError(ValueError):
    pass


def _fail(message: str):
    raise ConfigError(message)


def _keys(value, where: str, allowed: set[str]) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        _fail(f"{where} must be a mapping")
    unknown = sorted(set(value) - allowed)
    if unknown:
        _fail(f"{where}: unknown field(s) {', '.join(unknown)}")
    return value


@dataclass(frozen=True)
class Upstream:
    id: str
    name: str
    base_url: str
    private: bool = False
    enabled: bool = True
    reasoning_effort: str | None = None   # None = default for the host

    @property
    def key_env(self) -> str:
        return "IR_KEY_" + self.id.upper().replace("-", "_")

    @property
    def hosted(self) -> bool:
        host = (urlparse(self.base_url).hostname or "").lower()
        return any(host == h or host.endswith("." + h) for h in HOSTED_API_HOSTS)

    def effective_reasoning_effort(self) -> str:
        if self.reasoning_effort is not None:
            return self.reasoning_effort
        return "" if self.hosted else "none"


@dataclass(frozen=True)
class Target:
    upstream: str
    model: str


@dataclass(frozen=True)
class Route:
    name: str
    targets: tuple[Target, ...]


@dataclass(frozen=True)
class CustomRule:
    name: str
    pattern: str
    keywords: tuple[str, ...] = ()
    validator: str | None = None
    ignore_case: bool = False


@dataclass(frozen=True)
class Rules:
    builtin: dict = field(default_factory=dict)
    custom: tuple[CustomRule, ...] = ()
    exceptions: tuple[str, ...] = ()

    def builtin_enabled(self, kind: str) -> bool:
        return bool(self.builtin.get(kind, True))


@dataclass(frozen=True)
class Privacy:
    mode: str = "off"
    laya: bool = False
    rules: Rules = field(default_factory=Rules)


@dataclass(frozen=True)
class Config:
    privacy: Privacy
    upstreams: tuple[Upstream, ...]
    routes: tuple[Route, ...]

    def upstream(self, upstream_id: str) -> Upstream | None:
        return next((u for u in self.upstreams if u.id == upstream_id), None)

    def route(self, name: str | None) -> Route | None:
        return next((r for r in self.routes if r.name == name), None)

    def route_names(self) -> list[str]:
        return [r.name for r in self.routes]

    def live_targets(self, route: Route, private_only: bool = False) -> list[Target]:
        out = []
        for target in route.targets:
            up = self.upstream(target.upstream)
            if up and up.enabled and (up.private or not private_only):
                out.append(target)
        return out


def _parse_upstream(raw, i: int) -> Upstream:
    where = f"upstreams[{i}]"
    raw = _keys(raw, where, {"id", "name", "base_url", "private", "enabled", "reasoning_effort"})
    uid = str(raw.get("id") or "")
    if not _ID.match(uid):
        _fail(f"{where}.id must be lowercase letters, digits and dashes (max 32)")
    url = str(raw.get("base_url") or "").strip().rstrip("/")
    parsed = urlparse(url)
    if parsed.scheme not in ("https", "http") or not parsed.hostname or parsed.username or parsed.password \
            or parsed.query or parsed.fragment:
        _fail(f"{where}.base_url must be an http(s) URL without credentials, e.g. https://api.example.com/v1")
    if parsed.scheme == "http" and parsed.hostname not in ("127.0.0.1", "localhost") \
            and not parsed.hostname.startswith(("10.", "192.168.", "172.")):
        _fail(f"{where}.base_url: plain http is only allowed for local or private addresses")
    effort = raw.get("reasoning_effort")
    if effort is not None and str(effort) not in ("", "none", "low", "medium", "high"):
        _fail(f"{where}.reasoning_effort must be none, low, medium, high or empty")
    return Upstream(id=uid, name=str(raw.get("name") or uid), base_url=url,
                    private=bool(raw.get("private", False)), enabled=bool(raw.get("enabled", True)),
                    reasoning_effort=None if effort is None else str(effort))


def _parse_route(raw, i: int, upstream_ids: set[str]) -> Route:
    where = f"routes[{i}]"
    raw = _keys(raw, where, {"name", "targets"})
    name = str(raw.get("name") or "")
    if not _ROUTE.match(name):
        _fail(f"{where}.name must be a model name callers can use (letters, digits, . _ : / -)")
    targets = raw.get("targets") or []
    if not isinstance(targets, list) or not targets:
        _fail(f"{where}.targets must list at least one upstream and model")
    parsed = []
    for j, t in enumerate(targets):
        t = _keys(t, f"{where}.targets[{j}]", {"upstream", "model"})
        up, model = str(t.get("upstream") or ""), str(t.get("model") or "").strip()
        if up not in upstream_ids:
            _fail(f"{where}.targets[{j}].upstream '{up}' is not a configured upstream")
        if not model or len(model) > 200 or any(c.isspace() for c in model):
            _fail(f"{where}.targets[{j}].model must be the upstream's model name")
        parsed.append(Target(up, model))
    return Route(name=name, targets=tuple(parsed))


def _parse_rules(raw) -> Rules:
    raw = _keys(raw, "privacy.rules", {"builtin", "custom", "exceptions"})
    builtin = _keys(raw.get("builtin"), "privacy.rules.builtin", set(BUILTIN_TYPES))
    custom = []
    seen = set()
    for i, r in enumerate(raw.get("custom") or []):
        where = f"privacy.rules.custom[{i}]"
        r = _keys(r, where, {"name", "pattern", "keywords", "validator", "ignore_case"})
        name = str(r.get("name") or "")
        if not _RULE_NAME.match(name):
            _fail(f"{where}.name must be UPPER_CASE (it becomes the placeholder type, e.g. CLAIM_NUMBER)")
        if name in seen or name.lower() in BUILTIN_TYPES:
            _fail(f"{where}.name '{name}' is already used")
        seen.add(name)
        pattern = str(r.get("pattern") or "")
        if not pattern or len(pattern) > 500:
            _fail(f"{where}.pattern is required (max 500 characters)")
        try:
            import regex
            regex.compile(pattern)
        except Exception as error:  # noqa: BLE001
            _fail(f"{where}.pattern is not a valid regular expression: {error}")
        validator = r.get("validator")
        if validator is not None and validator not in VALIDATORS:
            _fail(f"{where}.validator must be one of {', '.join(VALIDATORS)}")
        keywords = r.get("keywords") or []
        if not isinstance(keywords, list) or any(not isinstance(k, str) or not k.strip() for k in keywords):
            _fail(f"{where}.keywords must be a list of words")
        custom.append(CustomRule(name=name, pattern=pattern, keywords=tuple(k.strip() for k in keywords),
                                 validator=validator, ignore_case=bool(r.get("ignore_case", False))))
    exceptions = raw.get("exceptions") or []
    if not isinstance(exceptions, list) or any(not isinstance(e, str) or not e.strip() for e in exceptions):
        _fail("privacy.rules.exceptions must be a list of values")
    return Rules(builtin={k: bool(v) for k, v in builtin.items()}, custom=tuple(custom),
                 exceptions=tuple(e.strip() for e in exceptions))


def parse(data) -> Config:
    data = _keys(data, "config", {"version", "privacy", "upstreams", "routes"})
    if data.get("version", 1) != 1:
        _fail("version must be 1")
    privacy_raw = _keys(data.get("privacy"), "privacy", {"mode", "laya", "rules"})
    mode = str(privacy_raw.get("mode") or "off")
    if mode not in PRIVACY_MODES:
        _fail(f"privacy.mode must be one of {', '.join(PRIVACY_MODES)}")
    privacy = Privacy(mode=mode, laya=bool(privacy_raw.get("laya", mode == "mask_check")),
                      rules=_parse_rules(privacy_raw.get("rules")))
    if mode == "mask_check" and not privacy.laya:
        _fail("privacy.mode mask_check needs privacy.laya: true")
    raw_ups = data.get("upstreams") or []
    if not isinstance(raw_ups, list):
        _fail("upstreams must be a list")
    upstreams = [_parse_upstream(u, i) for i, u in enumerate(raw_ups)]
    ids = [u.id for u in upstreams]
    if len(ids) != len(set(ids)):
        _fail("upstream ids must be unique")
    raw_routes = data.get("routes") or []
    if not isinstance(raw_routes, list):
        _fail("routes must be a list")
    routes = [_parse_route(r, i, set(ids)) for i, r in enumerate(raw_routes)]
    names = [r.name for r in routes]
    if len(names) != len(set(names)):
        _fail("route names must be unique")
    if mode == "keep_private" and not any(u.private and u.enabled for u in upstreams):
        _fail("privacy.mode keep_private needs at least one enabled upstream marked private")
    return Config(privacy=privacy, upstreams=tuple(upstreams), routes=tuple(routes))


def load(path: str | None = None) -> Config:
    path = path or os.environ.get("IR_CONFIG", DEFAULT_PATH)
    with open(path, encoding="utf-8") as fh:
        return parse(yaml.safe_load(fh) or {})


def dump(config_data: dict) -> str:
    """Validate then serialise; callers write the result atomically."""
    parse(config_data)
    return yaml.safe_dump(config_data, sort_keys=False, allow_unicode=True)
