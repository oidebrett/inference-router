"""Inference Router admin service: config, upstream keys and tests, the Experimenter, and a
live metadata feed. Serves the web UI from ./static.

    IR_ADMIN_HOST=127.0.0.1 IR_ADMIN_PORT=4110 python -m ir_admin.app

Runs as the inference-router user next to the router (same venv, same code). It writes
router.yaml and secrets.env; inference-router-reload.path restarts the router when they
change. Nothing here stores message content: the Experimenter answers in the response
only, and the feed holds metadata events in memory (gone on restart).

Personal subscriptions: a Claude setup token is saved like any upstream key; ChatGPT is
signed in here with the device-code flow (POST .../chatgpt/start, then poll .../chatgpt),
and the tokens go to CHATGPT_TOKEN_DIR, never to the browser.

Auth: the admin key (LITELLM_MASTER_KEY in secrets.env) as a Bearer token, or a session
cookie from POST /api/login. Put the whole service behind Pangolin SSO in production.
"""

from __future__ import annotations

import asyncio
import copy
import hmac
import json
import os
import secrets
import time
from collections import deque
from pathlib import Path

import httpx
import yaml
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from inference_router.config import ConfigError, parse
from inference_router.detection import DetectionTimeout, Detector
from inference_router.hook import RouterHook
from inference_router.keys import DEFAULT_RPM, KeyError_, KeyStore
from inference_router import render, subscriptions as subs

CONFIG_PATH = Path(os.environ.get("IR_CONFIG", "/etc/inference-router/router.yaml"))
SECRETS_PATH = Path(os.environ.get("IR_SECRETS", "/etc/inference-router/secrets.env"))
ROUTER_ENV_PATH = Path(os.environ.get("IR_ROUTER_ENV", "/etc/inference-router/router.env"))
STATIC = Path(__file__).parent / "static"
SESSION_TTL = 12 * 3600
EXPERIMENT_TIMEOUT = 60.0

app = FastAPI(title="Inference Router admin", docs_url=None, redoc_url=None, openapi_url=None)
_sessions: dict[str, float] = {}
_events: deque = deque(maxlen=500)
_listeners: set[asyncio.Queue] = set()
_tests: dict[str, dict] = {}   # upstream id -> last Test result (no content beyond a short reply)
_keystore: KeyStore | None = None
_chatgpt_login: dict = {}       # the one pending device-code sign-in (code, expiry, state; no tokens)
ADMIN_META = {"ir_key_name": "admin", "ir_subscriptions": True}


def keystore() -> KeyStore:
    global _keystore
    if _keystore is None:
        _keystore = KeyStore()
    return _keystore


def _laya_available(cfg) -> bool:
    return cfg.privacy.laya or _read_env(ROUTER_ENV_PATH).get("IR_LAYA_AVAILABLE") == "1"


# --- files ----------------------------------------------------------------------------

def _read_env(path: Path) -> dict:
    out = {}
    try:
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    except FileNotFoundError:
        pass
    return out


def _secrets() -> dict:
    return _read_env(SECRETS_PATH)


def _atomic_write(path: Path, text: str, mode: int) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.chmod(tmp, mode)
    os.replace(tmp, path)


def _write_secret(name: str, value: str) -> None:
    lines = []
    try:
        lines = [l for l in SECRETS_PATH.read_text().splitlines() if not l.startswith(name + "=")]
    except FileNotFoundError:
        pass
    lines.append(f"{name}={value}")
    _atomic_write(SECRETS_PATH, "\n".join(lines) + "\n", 0o600)


def _raw_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text()) or {}


def _config():
    return parse(_raw_config())


# --- auth -----------------------------------------------------------------------------

def _admin_key() -> str:
    return _secrets().get("LITELLM_MASTER_KEY", "")


def require_admin(request: Request) -> None:
    now = time.time()
    token = request.cookies.get("ir_session")
    if token and _sessions.get(token, 0) > now:
        return
    auth = request.headers.get("authorization", "")
    key = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    admin = _admin_key()
    if key and admin and hmac.compare_digest(key.encode(), admin.encode()):
        return
    raise HTTPException(status_code=401, detail="Sign in with the admin key")


@app.post("/api/login")
async def login(request: Request):
    body = await request.json()
    key = str(body.get("key") or "").strip()
    admin = _admin_key()
    if not key or not admin or not hmac.compare_digest(key.encode(), admin.encode()):
        await asyncio.sleep(1.0)
        raise HTTPException(status_code=401, detail="Wrong admin key")
    token = secrets.token_urlsafe(32)
    _sessions[token] = time.time() + SESSION_TTL
    resp = JSONResponse({"ok": True})
    resp.set_cookie("ir_session", token, httponly=True, samesite="strict", max_age=SESSION_TTL,
                    secure=request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https")
    return resp


@app.post("/api/logout")
async def logout(request: Request):
    _sessions.pop(request.cookies.get("ir_session", ""), None)
    resp = JSONResponse({"ok": True})
    resp.delete_cookie("ir_session")
    return resp


# --- config ---------------------------------------------------------------------------

@app.get("/api/config", dependencies=[Depends(require_admin)])
async def get_config():
    raw = _raw_config()
    s = _secrets()
    try:
        cfg = parse(raw)
        keys = {u.id: bool(s.get(u.key_env)) and s.get(u.key_env) != "none" for u in cfg.upstreams}
        error = None
    except ConfigError as e:
        keys, error = {}, str(e)
    return {"config": raw, "keys_set": keys, "error": error}


@app.put("/api/config", dependencies=[Depends(require_admin)])
async def put_config(request: Request):
    raw = await request.json()
    try:
        cfg = parse(raw)
    except ConfigError as e:
        raise HTTPException(status_code=400, detail=str(e))
    s = _secrets()
    for up in cfg.upstreams:   # every upstream needs a resolvable key entry
        if up.key_env not in s:
            _write_secret(up.key_env, "none")
    _atomic_write(CONFIG_PATH, yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), 0o640)
    return {"ok": True, "note": "Saved. The router restarts with the new configuration (a few seconds)."}


@app.put("/api/upstreams/{upstream_id}/key", dependencies=[Depends(require_admin)])
async def put_key(upstream_id: str, request: Request):
    body = await request.json()
    key = str(body.get("key") or "").strip()
    up = _config().upstream(upstream_id)
    if up is None:
        raise HTTPException(status_code=404, detail="Unknown upstream")
    if not key or any(c.isspace() for c in key) or len(key) > 500:
        raise HTTPException(status_code=400, detail="Enter the API key (no spaces)")
    if up.kind == subs.CHATGPT:
        raise HTTPException(status_code=400, detail="A ChatGPT subscription is connected with Sign in, not a key")
    if up.kind == subs.CLAUDE and not subs.is_claude_token(key):
        raise HTTPException(status_code=400, detail="Paste the token from 'claude setup-token' (it starts sk-ant-oat)")
    if up.kind != subs.CLAUDE and subs.is_claude_token(key):
        raise HTTPException(status_code=400, detail="That is a Claude subscription token: add it as a "
                                                    "'Claude subscription' upstream, not an API key")
    _write_secret(up.key_env, key)
    _tests.pop(upstream_id, None)
    return {"ok": True}


# --- upstream test --------------------------------------------------------------------

@app.post("/api/upstreams/{upstream_id}/test", dependencies=[Depends(require_admin)])
async def test_upstream(upstream_id: str, request: Request):
    body = await request.json() if (await request.body()) else {}
    cfg = _config()
    up = cfg.upstream(upstream_id)
    if up is None:
        raise HTTPException(status_code=404, detail="Unknown upstream")
    model = body.get("model") or next((t.model for r in cfg.routes for t in r.targets if t.upstream == up.id), None)
    if up.subscription:
        result = {"at": time.time(), "model": model}
        if not model:
            result.update(ok=False, error="Add this upstream to a route first (the test uses that model)")
        else:
            out = await _call_subscription(cfg, up, model, [{"role": "user", "content": "Reply with the single word OK."}], 256)
            result.update(ok="reply" in out, chat_status=200 if "reply" in out else None, **out)
        _tests[upstream_id] = result
        return result
    key = _secrets().get(up.key_env, "")
    headers = {"Authorization": f"Bearer {key}"} if key and key != "none" else {}
    result = {"at": time.time(), "model": model}
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            t = time.time()
            r = await client.get(f"{up.base_url}/models", headers=headers)
            result["models_status"] = r.status_code
            if r.status_code == 200:
                ids = [m.get("id") for m in (r.json().get("data") or []) if isinstance(m, dict)]
                result["model_count"] = len(ids)
                result["model_listed"] = model in ids if model else None
            if model:
                t = time.time()
                r = await client.post(f"{up.base_url}/chat/completions", headers=headers, json={
                    "model": model, "max_tokens": 256,
                    "messages": [{"role": "user", "content": "Reply with the single word OK."}]})
                result["chat_status"] = r.status_code
                result["latency_ms"] = round((time.time() - t) * 1000)
                if r.status_code == 200:
                    msg = (r.json().get("choices") or [{}])[0].get("message") or {}
                    result["reply"] = str(msg.get("content") or "")[:80]
                else:
                    result["error"] = r.text[:200]
            result["ok"] = result.get("chat_status", result.get("models_status")) == 200
        except httpx.HTTPError as e:
            result.update(ok=False, error=type(e).__name__)
    _tests[upstream_id] = result
    return result


async def _call_subscription(cfg, up, model: str, messages: list, max_tokens: int) -> dict:
    """One request to a subscription upstream, built exactly as the router builds it
    (render params + the Claude identity). Returns reply/latency_ms or error."""
    import litellm
    target = next((t for r in cfg.routes for t in r.targets if t.upstream == up.id and t.model == model), None)
    params = render._params(cfg, target or type("T", (), {"upstream": up.id, "model": model})())
    kwargs = {k: v for k, v in params.items() if k != "api_key"}
    if up.kind == subs.CLAUDE:
        token = _secrets().get(up.key_env, "")
        if not subs.is_claude_token(token):
            return {"error": "No Claude setup token saved for this upstream yet"}
        kwargs["api_key"] = token
        messages = subs.with_claude_identity(messages)
    else:
        os.environ.setdefault("CHATGPT_TOKEN_DIR", str(subs.chatgpt_dir(_read_env(ROUTER_ENV_PATH))))
        if not subs.chatgpt_signed_in():
            return {"error": "ChatGPT is not signed in yet"}
    subs.never_prompt_for_chatgpt_login()
    t = time.time()
    try:
        resp = await litellm.acompletion(messages=messages, max_tokens=max_tokens, timeout=EXPERIMENT_TIMEOUT,
                                         num_retries=0, **kwargs)
    except Exception as e:  # noqa: BLE001 — report the provider's answer, never the token
        status = getattr(e, "status_code", None)
        text = str(getattr(e, "message", "") or e)
        for secret in (kwargs.get("api_key"),):
            if secret:
                text = text.replace(secret, "***")
        return {"error": f"{status or type(e).__name__}: {text[:200]}", "status": status}
    msg = resp.choices[0].message
    return {"reply": str(msg.content or ""), "latency_ms": round((time.time() - t) * 1000)}


# --- ChatGPT sign-in --------------------------------------------------------------------

def _chatgpt_upstream():
    up = next((u for u in _config().upstreams if u.kind == subs.CHATGPT), None)
    if up is None:
        raise HTTPException(status_code=404, detail="Add a ChatGPT subscription upstream first")
    return up


def _chatgpt_dir() -> Path:
    return subs.chatgpt_dir(_read_env(ROUTER_ENV_PATH))


@app.get("/api/chatgpt", dependencies=[Depends(require_admin)])
async def chatgpt_state():
    pending = _chatgpt_login if _chatgpt_login.get("expires", 0) > time.time() else {}
    return {**subs.chatgpt_status(_chatgpt_dir()),
            "pending": {k: pending[k] for k in ("user_code", "verify_url", "state", "error") if k in pending} or None}


async def _poll_chatgpt(login: dict) -> None:
    async with httpx.AsyncClient(timeout=20.0) as client:
        while login is _chatgpt_login and login["expires"] > time.time():
            await asyncio.sleep(max(login["interval"], 5))
            try:
                tokens = await subs.poll_device_login(client, login)
            except (httpx.HTTPError, ValueError) as e:
                login.update(state="failed", error=f"Sign-in failed ({type(e).__name__}); start again")
                return
            if tokens:
                subs.save_chatgpt_tokens(_chatgpt_dir(), tokens)   # the stamp restarts the router
                login.update(state="done")
                _tests.pop(login["upstream"], None)
                return
        if login is _chatgpt_login and login.get("state") == "waiting":
            login.update(state="expired", error="The code expired; start again")


@app.post("/api/chatgpt/start", dependencies=[Depends(require_admin)])
async def chatgpt_start():
    global _chatgpt_login
    up = _chatgpt_upstream()
    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            pending = await subs.start_device_login(client)
        except (httpx.HTTPError, ValueError) as e:
            raise HTTPException(status_code=502, detail=f"OpenAI sign-in is not reachable ({type(e).__name__})")
    _chatgpt_login = {**pending, "state": "waiting", "expires": time.time() + 15 * 60, "upstream": up.id}
    asyncio.create_task(_poll_chatgpt(_chatgpt_login))
    return {"user_code": pending["user_code"], "verify_url": pending["verify_url"],
            "note": "Open the link, sign in to ChatGPT and enter the code. Never share this code."}


@app.post("/api/chatgpt/signout", dependencies=[Depends(require_admin)])
async def chatgpt_signout():
    global _chatgpt_login
    _chatgpt_login = {}
    subs.sign_out_chatgpt(_chatgpt_dir())
    for up in _config().upstreams:
        if up.kind == subs.CHATGPT:
            _tests.pop(up.id, None)
    return {"ok": True, "note": "Signed out. The router stops using ChatGPT after its restart (a few seconds)."}


# --- experimenter ---------------------------------------------------------------------

def _hook(cfg) -> RouterHook:
    env = {**_read_env(ROUTER_ENV_PATH), **_secrets()}
    return RouterHook(config=cfg, env=env)


def _spans(detector: Detector, messages: list) -> list:
    out = []
    for i, m in enumerate(messages):
        content = m.get("content") if isinstance(m, dict) else None
        if isinstance(content, str):
            try:
                found = [{"type": s.type, "start": s.start, "end": s.end} for s in detector.find_spans(content)]
            except DetectionTimeout as e:
                found = [{"type": f"rule_timeout:{e}", "start": 0, "end": len(content)}]
            out.append({"message": i, "spans": found})
    return out


@app.post("/api/experiment", dependencies=[Depends(require_admin)])
async def experiment(request: Request):
    """Dry run by default: the router's own decision code, nothing forwarded or stored.
    With send=true the masked request goes to the route's first target and both the raw
    reply (placeholders) and the restored reply are returned. Never logged."""
    body = await request.json()
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages or len(json.dumps(messages)) > 200_000:
        raise HTTPException(status_code=400, detail="Send messages: a non-empty list (max 200 KB)")
    cfg = _config()
    if body.get("mode"):
        raw = _raw_config()
        raw.setdefault("privacy", {})["mode"] = body["mode"]
        if body["mode"] in ("mask_check", "keep_private"):
            raw["privacy"]["laya"] = bool(raw["privacy"].get("laya", True)) or body["mode"] == "mask_check"
        try:
            cfg = parse(raw)
        except ConfigError as e:
            raise HTTPException(status_code=400, detail=str(e))
    route = body.get("route") or (cfg.route_names() or [None])[0]
    hook = _hook(cfg)
    hook._emit = lambda event: None   # the Experimenter is not traffic: no feed events
    data = {"model": route, "messages": copy.deepcopy(messages)}
    started = time.time()
    data = await hook.async_pre_call_hook(type("Admin", (), {"metadata": ADMIN_META})(), None, data, "acompletion")
    meta = hook._meta(data)
    event = meta.get("ir_event") or {}
    result = {
        "route": route, "mode": cfg.privacy.mode,
        "decision": meta.get("ir_decision"), "reason": meta.get("ir_reason"),
        "detected": _spans(hook.detector, messages),
        "masked_types": event.get("masked") or {}, "masked_values": event.get("masked_values", 0),
        "laya_max": event.get("laya_max"), "reasons": event.get("reasons") or [],
        "decide_ms": round((time.time() - started) * 1000, 1),
    }
    if result["decision"] == "blocked":
        result["refusal"] = data.get("mock_response")
        return result
    target_name = data["model"]
    chain = hook.chains.get(route, {})
    variant = "private" if result["decision"] == "private" else "all"
    names = chain.get(variant) or []
    targets = render.chain_targets(cfg, cfg.route(route), variant, hook.env)
    target = targets[names.index(target_name)] if target_name in names else None
    up = cfg.upstream(target.upstream) if target else None
    result["would_send"] = data["messages"]
    result["target"] = {"upstream": up.name if up else None, "model": target.model if target else None,
                        "fallbacks": [cfg.upstream(t.upstream).name + " · " + t.model for t in targets[1:]]}
    result["target"]["subscription"] = bool(up and up.subscription)
    if body.get("send") and up and up.subscription:
        out = await _call_subscription(cfg, up, target.model, data["messages"], 400)
        if "reply" in out:
            from pii_router.mask import restore_text
            result.update(upstream_ms=out["latency_ms"], raw_reply=out["reply"],
                          restored_reply=restore_text(out["reply"], hook._peek_map(data)))
        else:
            result["send_error"] = f"{up.name}: {out['error']}"
    elif body.get("send") and up:
        key = _secrets().get(up.key_env, "")
        headers = {"Authorization": f"Bearer {key}"} if key and key != "none" else {}
        payload = {"model": target.model, "messages": data["messages"], "max_tokens": 400}
        effort = up.effective_reasoning_effort()
        if effort:
            payload["reasoning_effort"] = effort
        try:
            async with httpx.AsyncClient(timeout=EXPERIMENT_TIMEOUT) as client:
                t = time.time()
                r = await client.post(f"{up.base_url}/chat/completions", headers=headers, json=payload)
            result["upstream_ms"] = round((time.time() - t) * 1000)
            if r.status_code != 200:
                result["send_error"] = f"{up.name} answered {r.status_code}: {r.text[:200]}"
            else:
                raw_text = str(((r.json().get("choices") or [{}])[0].get("message") or {}).get("content") or "")
                result["raw_reply"] = raw_text
                from pii_router.mask import restore_text
                result["restored_reply"] = restore_text(raw_text, hook._peek_map(data))
        except httpx.HTTPError as e:
            result["send_error"] = f"{up.name}: {type(e).__name__}"
    hook._drop_map(data)
    return result


# --- API keys -------------------------------------------------------------------------


def _key_fields(body: dict, cfg, creating: bool) -> dict:
    """Map the form to KeyStore fields. Omitted fields keep the defaults: all routes, the
    router's privacy, 60 requests/minute, no daily token limit, no expiry."""
    out = {}
    if creating or "name" in body:
        out["name"] = str(body.get("name") or "")
    if "routes" in body or creating:
        routes = body.get("routes")
        if routes is not None:
            unknown = [r for r in routes if not cfg.route(r)] if isinstance(routes, list) else ["?"]
            if unknown:
                raise HTTPException(status_code=400, detail=f"Unknown route(s): {', '.join(map(str, unknown))}")
        out["routes"] = routes
    if "privacy" in body or creating:
        privacy = body.get("privacy") or None
        if privacy == "mask_check" and not _laya_available(cfg):
            raise HTTPException(status_code=400, detail="Mask + check needs the Laya classifier, which is not installed")
        if privacy == "keep_private" and not any(u.private and u.enabled for u in cfg.upstreams):
            raise HTTPException(status_code=400, detail="Keep private needs an upstream marked private")
        out["privacy"] = privacy
    for field, default in (("rpm", DEFAULT_RPM), ("tpd", None)):
        if field in body:
            value = body[field]
            out[field] = None if value in (None, "", 0) else int(value)
        elif creating:
            out[field] = default
    if "expires_days" in body:
        value = body["expires_days"]
        out["expires_days"] = None if value in (None, "", 0) else float(value)
    if "subscriptions" in body or creating:
        out["subscriptions"] = bool(body.get("subscriptions", False))
    return out


@app.get("/api/keys", dependencies=[Depends(require_admin)])
async def list_keys():
    cfg = _config()
    return {"keys": keystore().list(), "routes": cfg.route_names(), "laya_available": _laya_available(cfg),
            "has_private": any(u.private and u.enabled for u in cfg.upstreams), "default_rpm": DEFAULT_RPM,
            "has_subscription": any(u.subscription and u.enabled for u in cfg.upstreams)}


@app.post("/api/keys", dependencies=[Depends(require_admin)])
async def create_key(request: Request):
    body = await request.json()
    try:
        fields = _key_fields(body, _config(), creating=True)
        key, secret = keystore().create(**fields)
    except (KeyError_, ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"key": key.public(), "secret": secret,
            "note": "Copy this key now: it is shown only once and stored only as a hash."}


@app.patch("/api/keys/{key_id}", dependencies=[Depends(require_admin)])
async def update_key(key_id: int, request: Request):
    body = await request.json()
    try:
        key = keystore().update(key_id, **_key_fields(body, _config(), creating=False))
    except (KeyError_, ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"key": key.public()}


@app.post("/api/keys/{key_id}/revoke", dependencies=[Depends(require_admin)])
async def revoke_key(key_id: int):
    if not keystore().revoke(key_id):
        raise HTTPException(status_code=404, detail="No such active key")
    return {"ok": True}


# --- status and live feed ---------------------------------------------------------------

@app.post("/internal/events")
async def post_event(request: Request):
    token = request.headers.get("authorization", "")[7:]
    expected = _secrets().get("IR_EVENTS_TOKEN", "")
    if not expected or not hmac.compare_digest(token.encode(), expected.encode()):
        raise HTTPException(status_code=401)
    event = await request.json()
    allowed = {"ts", "route", "mode", "stream", "decision", "reason", "masked", "masked_values", "laya_max",
               "reasons", "target", "decide_ms", "outcome", "total_ms", "served_by", "error", "key", "tokens"}
    event = {k: v for k, v in event.items() if k in allowed}   # metadata only, whatever is sent
    _events.append(event)
    for q in list(_listeners):
        q.put_nowait(event)
    return Response(status_code=204)


@app.get("/api/events", dependencies=[Depends(require_admin)])
async def events_stream():
    queue: asyncio.Queue = asyncio.Queue()
    _listeners.add(queue)

    async def gen():
        try:
            for e in list(_events)[-50:]:
                yield f"data: {json.dumps(e)}\n\n"
            while True:
                try:
                    e = await asyncio.wait_for(queue.get(), timeout=15)
                    yield f"data: {json.dumps(e)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        finally:
            _listeners.discard(queue)
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/status", dependencies=[Depends(require_admin)])
async def status():
    try:
        cfg = _config()
        error = None
    except ConfigError as e:
        return {"error": str(e)}
    env = _read_env(ROUTER_ENV_PATH)
    router_url = f"http://{env.get('IR_HOST', '127.0.0.1').replace('0.0.0.0', '127.0.0.1')}:{env.get('IR_PORT', '4100')}"
    async with httpx.AsyncClient(timeout=3.0) as client:
        try:
            healthy = (await client.get(f"{router_url}/health/liveliness")).status_code == 200
        except httpx.HTTPError:
            healthy = False
    recent = [e for e in _events if e.get("ts", 0) > time.time() - 3600]
    counts = {"requests": len(recent), "masked": sum(1 for e in recent if e.get("reason") == "masked"),
              "blocked": sum(1 for e in recent if e.get("decision") == "blocked"),
              "private": sum(1 for e in recent if e.get("decision") == "private"),
              "errors": sum(1 for e in recent if e.get("outcome") == "error")}
    s = _secrets()
    upstreams = []
    for up in cfg.upstreams:
        key_set = bool(s.get(up.key_env)) and s.get(up.key_env) != "none"
        extra = {}
        if up.kind == subs.CHATGPT:
            extra["chatgpt"] = subs.chatgpt_status(_chatgpt_dir())
            key_set = extra["chatgpt"]["signed_in"]
        upstreams.append({"id": up.id, "name": up.name, "base_url": up.base_url, "private": up.private,
                          "enabled": up.enabled, "hosted": up.hosted, "kind": up.kind,
                          "subscription": up.subscription, **extra,
                          "key_set": key_set,
                          "models": sorted({t.model for r in cfg.routes for t in r.targets if t.upstream == up.id}),
                          "test": _tests.get(up.id)})
    return {"router": {"healthy": healthy, "url": os.environ.get("IR_PUBLIC_URL") or f"{router_url}/v1"},
            "privacy": {"mode": cfg.privacy.mode, "laya": cfg.privacy.laya,
                        "custom_rules": len(cfg.privacy.rules.custom)},
            "routes": [{"name": r.name, "targets": [{"upstream": t.upstream, "model": t.model} for t in r.targets]}
                       for r in cfg.routes],
            "upstreams": upstreams, "last_hour": counts, "error": error}


# --- UI -------------------------------------------------------------------------------

if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


def main():
    import uvicorn
    uvicorn.run(app, host=os.environ.get("IR_ADMIN_HOST", "127.0.0.1"),
                port=int(os.environ.get("IR_ADMIN_PORT", "4110")), log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
