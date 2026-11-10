"""Admin service: auth, config validation and save, upstream keys, the Experimenter and the
event intake (metadata only)."""

import importlib
import json

import pytest
import yaml
from fastapi.testclient import TestClient

CONFIG = {
    "version": 1,
    "privacy": {"mode": "mask", "rules": {"custom": [{"name": "CLAIM_NUMBER", "pattern": r"CLM-\d{4}-\d{7}"}]}},
    "upstreams": [{"id": "echo", "name": "Echo", "base_url": "http://127.0.0.1:9/v1"}],
    "routes": [{"name": "default", "targets": [{"upstream": "echo", "model": "m"}]}],
}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    (tmp_path / "router.yaml").write_text(yaml.safe_dump(CONFIG))
    (tmp_path / "secrets.env").write_text("LITELLM_MASTER_KEY=sk-admin\nIR_MASK_SECRET=s\nIR_EVENTS_TOKEN=evt\n")
    (tmp_path / "router.env").write_text("IR_HOST=127.0.0.1\nIR_PORT=9\n")
    monkeypatch.setenv("IR_CONFIG", str(tmp_path / "router.yaml"))
    monkeypatch.setenv("IR_SECRETS", str(tmp_path / "secrets.env"))
    monkeypatch.setenv("IR_ROUTER_ENV", str(tmp_path / "router.env"))
    monkeypatch.setenv("IR_KEYS_DB", str(tmp_path / "keys.db"))
    import ir_admin.app as app_module
    app_module = importlib.reload(app_module)
    c = TestClient(app_module.app)
    c.tmp = tmp_path
    return c


AUTH = {"Authorization": "Bearer sk-admin"}


def test_auth_required_and_login(client):
    assert client.get("/api/config").status_code == 401
    assert client.post("/api/login", json={"key": "wrong"}).status_code == 401
    assert client.post("/api/login", json={"key": "sk-admin"}).status_code == 200
    assert client.get("/api/config").status_code == 200          # session cookie
    assert client.get("/api/config", headers=AUTH).status_code == 200


def test_invalid_config_is_refused_and_file_unchanged(client):
    before = (client.tmp / "router.yaml").read_text()
    bad = json.loads(json.dumps(CONFIG)); bad["privacy"]["mode"] = "encrypt"
    r = client.put("/api/config", json=bad, headers=AUTH)
    assert r.status_code == 400 and "privacy.mode" in r.json()["detail"]
    assert (client.tmp / "router.yaml").read_text() == before


def test_save_config_adds_key_entries_and_key_is_never_returned(client):
    new = json.loads(json.dumps(CONFIG))
    new["upstreams"].append({"id": "openrouter", "name": "OpenRouter", "base_url": "https://openrouter.ai/api/v1"})
    assert client.put("/api/config", json=new, headers=AUTH).status_code == 200
    assert "IR_KEY_OPENROUTER=none" in (client.tmp / "secrets.env").read_text()
    assert client.put("/api/upstreams/openrouter/key", json={"key": "sk-or-secret"}, headers=AUTH).status_code == 200
    assert "IR_KEY_OPENROUTER=sk-or-secret" in (client.tmp / "secrets.env").read_text()
    body = client.get("/api/config", headers=AUTH).text
    assert "sk-or-secret" not in body and json.loads(body)["keys_set"]["openrouter"] is True


def test_experiment_dry_run_masks_and_never_calls_upstreams(client):
    r = client.post("/api/experiment", headers=AUTH, json={
        "route": "default", "messages": [{"role": "user", "content": "claim CLM-2026-0048213 for ann@acme.ie"}]})
    d = r.json()
    assert d["decision"] == "upstream" and d["reason"] == "masked"
    assert d["masked_types"] == {"claim_number": 1, "email": 1}
    sent = json.dumps(d["would_send"])
    assert "CLM-2026" not in sent and "ann@acme.ie" not in sent
    assert {s["type"] for s in d["detected"][0]["spans"]} == {"claim_number", "email"}
    assert "raw_reply" not in d


def test_events_need_the_token_and_keep_metadata_only(client):
    ev = {"ts": 1, "route": "default", "decision": "upstream", "content": "ann@acme.ie", "messages": ["x"]}
    assert client.post("/internal/events", json=ev).status_code == 401
    assert client.post("/internal/events", json=ev, headers={"Authorization": "Bearer evt"}).status_code == 204
    import ir_admin.app as app_module
    stored = list(app_module._events)[-1]
    assert "content" not in stored and "messages" not in stored and stored["route"] == "default"


def test_keys_create_with_only_a_name_list_and_revoke(client):
    r = client.post("/api/keys", headers=AUTH, json={"name": "Ann (finance)"})
    assert r.status_code == 200
    body = r.json()
    assert body["secret"].startswith("sk-ir-")
    assert body["key"]["routes"] is None and body["key"]["rpm"] == 60 and body["key"]["privacy"] is None
    listed = client.get("/api/keys", headers=AUTH).json()
    assert listed["routes"] == ["default"] and body["secret"] not in json.dumps(listed)
    key_id = body["key"]["id"]
    assert client.post(f"/api/keys/{key_id}/revoke", headers=AUTH).status_code == 200
    assert client.post(f"/api/keys/{key_id}/revoke", headers=AUTH).status_code == 404


def test_keys_validation(client):
    assert client.post("/api/keys", headers=AUTH, json={"name": ""}).status_code == 400
    assert client.post("/api/keys", headers=AUTH, json={"name": "x", "routes": ["ghost"]}).status_code == 400
    r = client.post("/api/keys", headers=AUTH, json={"name": "x", "privacy": "mask_check"})
    assert r.status_code == 400 and "Laya" in r.json()["detail"]          # not installed in this test
    r = client.post("/api/keys", headers=AUTH, json={"name": "x", "privacy": "keep_private"})
    assert r.status_code == 400 and "private" in r.json()["detail"]
    r = client.post("/api/keys", headers=AUTH, json={"name": "x", "rpm": None, "tpd": 5000, "expires_days": 30})
    assert r.status_code == 200 and r.json()["key"]["rpm"] is None and r.json()["key"]["tpd"] == 5000


# --- personal subscriptions -------------------------------------------------------------

def _with_subscriptions(client):
    (client.tmp / "router.env").write_text(f"IR_HOST=127.0.0.1\nIR_PORT=9\nCHATGPT_TOKEN_DIR={client.tmp / 'chatgpt'}\n")
    new = json.loads(json.dumps(CONFIG))
    new["upstreams"] += [{"id": "my-claude", "name": "My Claude", "kind": "claude_subscription"},
                         {"id": "my-chatgpt", "name": "My ChatGPT", "kind": "chatgpt_subscription"}]
    new["routes"].append({"name": "claude", "targets": [{"upstream": "my-claude", "model": "claude-sonnet-4-6"},
                                                        {"upstream": "echo", "model": "m"}]})
    assert client.put("/api/config", json=new, headers=AUTH).status_code == 200


def test_subscription_tokens_are_checked_where_they_go(client):
    _with_subscriptions(client)
    r = client.put("/api/upstreams/my-claude/key", json={"key": "sk-ant-api03-x"}, headers=AUTH)
    assert r.status_code == 400 and "setup-token" in r.json()["detail"]
    assert client.put("/api/upstreams/my-claude/key", json={"key": "sk-ant-oat01-x"}, headers=AUTH).status_code == 200
    r = client.put("/api/upstreams/echo/key", json={"key": "sk-ant-oat01-x"}, headers=AUTH)
    assert r.status_code == 400 and "Claude subscription" in r.json()["detail"]
    assert client.put("/api/upstreams/my-chatgpt/key", json={"key": "x"}, headers=AUTH).status_code == 400
    ups = {u["id"]: u for u in client.get("/api/status", headers=AUTH).json()["upstreams"]}
    assert ups["my-claude"]["subscription"] and ups["my-claude"]["key_set"]
    assert ups["my-chatgpt"]["chatgpt"] == {"signed_in": False} and not ups["my-chatgpt"]["key_set"]
    assert "sk-ant-oat01-x" not in json.dumps(client.get("/api/config", headers=AUTH).json())


def test_chatgpt_sign_in_flow_never_exposes_tokens(client, monkeypatch):
    _with_subscriptions(client)
    import ir_admin.app as app_module
    from inference_router import subscriptions as subs

    async def fake_start(c):
        return {"user_code": "ABCD-1234", "device_auth_id": "dev", "interval": 0, "verify_url": subs.CHATGPT_VERIFY_URL}

    async def fake_poll(c, pending):
        return {"access_token": "h.e30.s", "refresh_token": "SECRET-REFRESH", "id_token": "h.eyJlbWFpbCI6ImFAYi5jIn0.s"}

    async def no_sleep(_):
        return None
    monkeypatch.setattr(app_module.subs, "start_device_login", fake_start)
    monkeypatch.setattr(app_module.subs, "poll_device_login", fake_poll)
    r = client.post("/api/chatgpt/start", headers=AUTH)
    assert r.status_code == 200 and r.json()["user_code"] == "ABCD-1234"
    import asyncio
    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    asyncio.run(app_module._poll_chatgpt(app_module._chatgpt_login))
    state = client.get("/api/chatgpt", headers=AUTH).json()
    assert state["signed_in"] and state["email"] == "a@b.c" and "SECRET-REFRESH" not in json.dumps(state)
    assert (client.tmp / "chatgpt" / "signed-in").exists()
    assert client.post("/api/chatgpt/signout", headers=AUTH).status_code == 200
    assert client.get("/api/chatgpt", headers=AUTH).json()["signed_in"] is False


def test_key_subscription_toggle(client):
    _with_subscriptions(client)
    assert client.get("/api/keys", headers=AUTH).json()["has_subscription"] is True
    k = client.post("/api/keys", json={"name": "My laptop", "subscriptions": True}, headers=AUTH).json()["key"]
    assert k["subscriptions"] is True
    assert client.post("/api/keys", json={"name": "Bob"}, headers=AUTH).json()["key"]["subscriptions"] is False
    assert client.patch(f"/api/keys/{k['id']}", json={"subscriptions": False}, headers=AUTH).json()["key"]["subscriptions"] is False


def test_experiment_dry_run_on_subscription_route_uses_admin_chain(client):
    _with_subscriptions(client)
    r = client.post("/api/experiment", json={"route": "claude", "messages": [{"role": "user", "content": "hi"}]},
                    headers=AUTH).json()
    assert r["target"]["upstream"] == "My Claude" and r["target"]["subscription"] is True
    assert r["target"]["fallbacks"] == ["Echo · m"]
