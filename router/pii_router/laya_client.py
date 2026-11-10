"""HTTP client for a local `laya-serve` (pip `laya[serve]`).

Any failure — connection refused, timeout, non-200, malformed body — raises,
and the policy turns that into a LOCAL route. Never return a default score.
"""

from __future__ import annotations

import os

import httpx

from .policy import PII_QUESTION


class LayaHttpScorer:
    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 15.0,
                 model: str | None = "english"):
        self.base_url = base_url.rstrip("/")
        # "english" keeps one ~421M checkpoint resident (RAM on an 8 GB box). Set
        # PII_ROUTER_LAYA_MODEL=multilingual for non-English traffic, or "" to let
        # Laya route per request (loads both checkpoints).
        self.model = model or None
        self.timeout = timeout
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(timeout=timeout)

    @classmethod
    def from_env(cls, env=os.environ) -> "LayaHttpScorer":
        return cls(
            base_url=env.get("PII_ROUTER_LAYA_URL", "http://127.0.0.1:4101"),
            api_key=env.get("LAYA_API_KEY") or None,
            timeout=float(env.get("PII_ROUTER_LAYA_TIMEOUT", "15")),
            model=env.get("PII_ROUTER_LAYA_MODEL", "english"),
        )

    def score(self, texts: list[str]) -> list[float]:
        if not texts:
            return []
        body = {"states": texts, "questions": {"pii": PII_QUESTION}}
        if self.model:
            body["model"] = self.model
        response = self._client.post(
            f"{self.base_url}/v1/systemone/batch",
            json=body,
            headers=self.headers,
        )
        response.raise_for_status()
        results = response.json()["results"]
        if len(results) != len(texts):
            raise ValueError(f"laya returned {len(results)} results for {len(texts)} states")
        return [float(item["answers"]["pii"]["noul"]) for item in results]
