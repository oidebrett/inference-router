"""Is the cloud backend (e.g. the customer's Vast GPU) up right now?

Used only when a health URL is configured (PrivateInference: GPU first, NVIDIA
as fallback). AgentGateway's NVIDIA-only setup does not set one and behaves as
before: cloud is always assumed up.

The probe result is cached for ``interval`` seconds, so at most one request per
interval pays for a probe. A failed cloud call marks the backend down at once
(LiteLLM retries are off, so that request fails; the next one is re-routed).
"""

from __future__ import annotations

import threading
import time
import urllib.request


class CloudHealth:
    def __init__(self, url: str, api_key: str = "", interval: float = 15.0, timeout: float = 3.0,
                 probe=None, clock=time.monotonic):
        self.url = url
        self.api_key = api_key
        self.interval = interval
        self.timeout = timeout
        self._probe = probe or self._http_probe
        self._clock = clock
        self._lock = threading.Lock()
        self._checked_at: float | None = None
        self._healthy = False

    @classmethod
    def from_env(cls, env) -> "CloudHealth | None":
        url = (env.get("PII_ROUTER_CLOUD_HEALTH_URL") or "").strip()
        if not url and (env.get("FALLBACK_BASE_URL") or "").strip() and (env.get("CLOUD_BASE_URL") or "").strip():
            url = env["CLOUD_BASE_URL"].rstrip("/") + "/models"
        if not url:
            return None
        try:
            interval = float(env.get("PII_ROUTER_CLOUD_HEALTH_INTERVAL") or 15)
        except ValueError:
            interval = 15.0
        return cls(url, api_key=env.get("CLOUD_API_KEY", ""), interval=interval)

    def _http_probe(self) -> bool:
        request = urllib.request.Request(self.url, headers={"Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310 — configured URL
                return 200 <= response.status < 300
        except Exception:  # noqa: BLE001 — any failure means "not up"
            return False

    def healthy(self) -> bool:
        with self._lock:
            now = self._clock()
            if self._checked_at is not None and now - self._checked_at < self.interval:
                return self._healthy
            self._checked_at = now  # claim the probe so concurrent callers use the cached value
        result = bool(self._probe())
        with self._lock:
            self._healthy = result
        return result

    def mark_down(self) -> None:
        with self._lock:
            self._healthy = False
            self._checked_at = self._clock()
