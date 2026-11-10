"""Per-person / per-team / per-app API keys, in one SQLite file.

The router (auth + usage) and the admin service (create, revoke, edit) share the file
(default /var/lib/inference-router/keys.db, WAL mode). Keys are stored as SHA-256
hashes; the secret is shown once when created. Usage is counts only: requests, tokens,
masked, blocked — never content.

Defaults make a key need only a name: all routes, the router's privacy setting,
60 requests per minute, no daily token limit, no expiry, and no use of the admin's
personal subscriptions (the admin turns that on for their own keys).
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass

DEFAULT_PATH = "/var/lib/inference-router/keys.db"
DEFAULT_RPM = 60
KEY_PREFIX = "sk-ir-"
PRIVACY_MODES = ("off", "mask", "mask_check", "keep_private")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS keys (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  prefix TEXT NOT NULL,
  hash TEXT NOT NULL UNIQUE,
  created REAL NOT NULL,
  expires REAL,
  revoked REAL,
  routes TEXT,              -- JSON list; NULL = all routes
  privacy TEXT,             -- NULL = the router's setting
  rpm INTEGER,              -- requests per minute; NULL = no limit
  tpd INTEGER,              -- tokens per day; NULL = no limit
  last_used REAL,
  subscriptions INTEGER NOT NULL DEFAULT 0   -- may use the admin's personal subscriptions
);
CREATE TABLE IF NOT EXISTS usage (
  key_id INTEGER NOT NULL,
  day TEXT NOT NULL,        -- UTC date
  requests INTEGER NOT NULL DEFAULT 0,
  tokens INTEGER NOT NULL DEFAULT 0,
  masked INTEGER NOT NULL DEFAULT 0,
  blocked INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (key_id, day)
);
"""


def _hash(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def _today(ts: float | None = None) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(ts or time.time()))


@dataclass(frozen=True)
class Key:
    id: int
    name: str
    prefix: str
    created: float
    expires: float | None
    revoked: float | None
    routes: tuple[str, ...] | None
    privacy: str | None
    rpm: int | None
    tpd: int | None
    last_used: float | None
    subscriptions: bool = False

    @property
    def active(self) -> bool:
        return self.revoked is None and (self.expires is None or self.expires > time.time())

    def allows(self, route: str) -> bool:
        return self.routes is None or route in self.routes

    def public(self) -> dict:
        return {"id": self.id, "name": self.name, "prefix": self.prefix, "created": self.created,
                "expires": self.expires, "revoked": self.revoked, "active": self.active,
                "routes": list(self.routes) if self.routes is not None else None, "privacy": self.privacy,
                "rpm": self.rpm, "tpd": self.tpd, "last_used": self.last_used, "subscriptions": self.subscriptions}


class KeyError_(ValueError):
    pass


class KeyStore:
    def __init__(self, path: str | None = None):
        self.path = path or os.environ.get("IR_KEYS_DB", DEFAULT_PATH)
        self._lock = threading.Lock()
        self._windows: dict[int, deque] = defaultdict(deque)
        with self._conn() as db:
            db.executescript(_SCHEMA)
            columns = {r["name"] for r in db.execute("PRAGMA table_info(keys)")}
            if "subscriptions" not in columns:   # keys.db from v0.1.0
                db.execute("ALTER TABLE keys ADD COLUMN subscriptions INTEGER NOT NULL DEFAULT 0")

    def _conn(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.execute("PRAGMA journal_mode=WAL")
        db.row_factory = sqlite3.Row
        return db

    @staticmethod
    def _row(r) -> Key:
        return Key(id=r["id"], name=r["name"], prefix=r["prefix"], created=r["created"], expires=r["expires"],
                   revoked=r["revoked"], routes=tuple(json.loads(r["routes"])) if r["routes"] else None,
                   privacy=r["privacy"], rpm=r["rpm"], tpd=r["tpd"], last_used=r["last_used"],
                   subscriptions=bool(r["subscriptions"]))

    # --- admin side ---------------------------------------------------------------

    @staticmethod
    def _validate(name, routes, privacy, rpm, tpd, expires_days, subscriptions=False):
        name = (name or "").strip()
        if not name or len(name) > 80:
            raise KeyError_("Give the key a name (who or what uses it), up to 80 characters")
        if routes is not None and (not isinstance(routes, list) or not all(isinstance(r, str) and r for r in routes)):
            raise KeyError_("routes must be a list of route names, or omitted for all routes")
        if privacy is not None and privacy not in PRIVACY_MODES:
            raise KeyError_(f"privacy must be one of {', '.join(PRIVACY_MODES)}, or omitted for the router's setting")
        for label, value in (("rpm", rpm), ("tpd", tpd)):
            if value is not None and (not isinstance(value, int) or value < 1):
                raise KeyError_(f"{label} must be a positive whole number, or omitted for no limit")
        if expires_days is not None and (not isinstance(expires_days, (int, float)) or expires_days <= 0):
            raise KeyError_("expires_days must be positive, or omitted for no expiry")
        if not isinstance(subscriptions, bool):
            raise KeyError_("subscriptions must be true or false")
        return name

    def create(self, name, routes=None, privacy=None, rpm=DEFAULT_RPM, tpd=None, expires_days=None,
               subscriptions=False) -> tuple[Key, str]:
        name = self._validate(name, routes, privacy, rpm, tpd, expires_days, subscriptions)
        secret = KEY_PREFIX + secrets.token_urlsafe(32)
        now = time.time()
        with self._lock, self._conn() as db:
            cur = db.execute(
                "INSERT INTO keys (name, prefix, hash, created, expires, routes, privacy, rpm, tpd, subscriptions) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (name, secret[:12], _hash(secret), now, now + expires_days * 86400 if expires_days else None,
                 json.dumps(routes) if routes is not None else None, privacy, rpm, tpd, int(subscriptions)))
            key = self._row(db.execute("SELECT * FROM keys WHERE id=?", (cur.lastrowid,)).fetchone())
        return key, secret

    def update(self, key_id: int, **fields) -> Key:
        current = self.get(key_id)
        if current is None:
            raise KeyError_("No such key")
        merged = {"name": current.name, "routes": list(current.routes) if current.routes is not None else None,
                  "privacy": current.privacy, "rpm": current.rpm, "tpd": current.tpd, "expires_days": None,
                  "subscriptions": current.subscriptions}
        merged.update({k: v for k, v in fields.items() if k in merged})
        self._validate(**merged)
        expires = current.expires
        if "expires_days" in fields:
            expires = time.time() + fields["expires_days"] * 86400 if fields["expires_days"] else None
        with self._lock, self._conn() as db:
            db.execute("UPDATE keys SET name=?, routes=?, privacy=?, rpm=?, tpd=?, expires=?, subscriptions=? WHERE id=?",
                       (merged["name"].strip(), json.dumps(merged["routes"]) if merged["routes"] is not None else None,
                        merged["privacy"], merged["rpm"], merged["tpd"], expires, int(merged["subscriptions"]), key_id))
        return self.get(key_id)

    def revoke(self, key_id: int) -> bool:
        with self._lock, self._conn() as db:
            return db.execute("UPDATE keys SET revoked=? WHERE id=? AND revoked IS NULL", (time.time(), key_id)).rowcount > 0

    def get(self, key_id: int) -> Key | None:
        with self._conn() as db:
            r = db.execute("SELECT * FROM keys WHERE id=?", (key_id,)).fetchone()
        return self._row(r) if r else None

    def list(self) -> list[dict]:
        today = _today()
        since = _today(time.time() - 29 * 86400)
        with self._conn() as db:
            rows = db.execute("SELECT * FROM keys ORDER BY revoked IS NOT NULL, created DESC").fetchall()
            usage = {}
            for u in db.execute("SELECT key_id, day, requests, tokens, masked, blocked FROM usage WHERE day >= ?", (since,)):
                agg = usage.setdefault(u["key_id"], {"today": {"requests": 0, "tokens": 0, "masked": 0, "blocked": 0},
                                                     "30d": {"requests": 0, "tokens": 0, "masked": 0, "blocked": 0}})
                for f in ("requests", "tokens", "masked", "blocked"):
                    agg["30d"][f] += u[f]
                    if u["day"] == today:
                        agg["today"][f] += u[f]
        out = []
        for r in rows:
            item = self._row(r).public()
            item["usage"] = usage.get(r["id"], {"today": {"requests": 0, "tokens": 0, "masked": 0, "blocked": 0},
                                                "30d": {"requests": 0, "tokens": 0, "masked": 0, "blocked": 0}})
            out.append(item)
        return out

    # --- router side ----------------------------------------------------------------

    def verify(self, secret: str) -> Key | None:
        if not secret or not secret.startswith(KEY_PREFIX):
            return None
        with self._conn() as db:
            r = db.execute("SELECT * FROM keys WHERE hash=?", (_hash(secret),)).fetchone()
        return self._row(r) if r else None

    def allow_request(self, key: Key, now: float | None = None) -> str | None:
        """None if the request may go ahead, else the reason it may not (rate or daily tokens)."""
        now = now or time.time()
        if key.rpm:
            with self._lock:
                window = self._windows[key.id]
                while window and window[0] <= now - 60:
                    window.popleft()
                if len(window) >= key.rpm:
                    return f"rate limit: {key.rpm} requests per minute"
                window.append(now)
        if key.tpd:
            with self._conn() as db:
                r = db.execute("SELECT tokens FROM usage WHERE key_id=? AND day=?", (key.id, _today(now))).fetchone()
            if r and r["tokens"] >= key.tpd:
                return f"daily token limit reached ({key.tpd})"
        return None

    def record(self, key_id: int, tokens: int = 0, masked: bool = False, blocked: bool = False) -> None:
        now = time.time()
        with self._lock, self._conn() as db:
            db.execute("INSERT INTO usage (key_id, day, requests, tokens, masked, blocked) VALUES (?,?,1,?,?,?) "
                       "ON CONFLICT(key_id, day) DO UPDATE SET requests=requests+1, tokens=tokens+excluded.tokens, "
                       "masked=masked+excluded.masked, blocked=blocked+excluded.blocked",
                       (key_id, _today(now), int(tokens or 0), int(masked), int(blocked)))
            db.execute("UPDATE keys SET last_used=? WHERE id=?", (now, key_id))
