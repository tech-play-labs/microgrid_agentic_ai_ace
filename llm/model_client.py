"""Backend-agnostic LLM access. Add a backend = subclass LLMClient + @register_backend("name") + a profile in config/model.json."""
import json, os, time
from abc import ABC, abstractmethod
from dataclasses import dataclass
import requests

class LLMError(Exception): pass
class LLMHTTPError(LLMError):
    def __init__(self, status, body): super().__init__(f"HTTP {status}: {body}"); self.status = status

@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    raw: dict | None = None

class LLMClient(ABC):
    @abstractmethod
    def complete(self, system: str, user: str, schema: dict, num_ctx: int, max_tokens: int) -> LLMResult:
        """Return raw text that should be a JSON object matching `schema`. Validation is the caller's job."""

_BACKENDS = {}
def register_backend(name):
    def deco(cls): _BACKENDS[name] = cls; return cls
    return deco

def _sleep(s): time.sleep(s)

def _post(url, headers, payload, timeout, retries=3):
    last = None
    for i in range(retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except requests.ReadTimeout as e:                      # the server took the request but did not answer in time
            raise LLMError(f"timeout after {timeout}s (not retried): {e}")
        except requests.RequestException as e:                 # connection problems are still retried
            last = LLMError(f"network error: {e}")
        else:
            if r.status_code == 200: return r.json()
            if r.status_code in (429, 500, 502, 503, 504): last = LLMHTTPError(r.status_code, r.text[:200])
            else: raise LLMHTTPError(r.status_code, r.text[:300])
        if i < retries - 1: _sleep(min(2 ** i, 8))
    raise last

@register_backend("ollama")
class OllamaClient(LLMClient):
    """Native Ollama /api/chat; the JSON schema goes in `format` (constrained decoding)."""
    supports_num_ctx = True

    def __init__(self, model, host="http://localhost:11434", temperature=0.0, seed=None,
                 max_tokens=300, timeout_s=120, extra_body=None):
        self.model, self.host, self.timeout = model, host.rstrip("/"), timeout_s
        self.options = {"temperature": temperature, "num_predict": max_tokens}
        if seed is not None: self.options["seed"] = seed
        self.max_tokens = max_tokens
        self.extra = extra_body or {}

    def complete(self, system, user, schema, num_ctx=None, max_tokens=None):
        options = dict(self.options)  # temperature, seed, etc. from the profile
        options["num_predict"] = max_tokens or self.max_tokens # build the options per call
        if num_ctx:
            options["num_ctx"] = num_ctx
        payload = {"model": self.model, "stream": False, "format": schema, "options": options,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], **self.extra}
        t0 = time.time(); d = _post(f"{self.host}/api/chat", {}, payload, self.timeout)
        return LLMResult(d.get("message", {}).get("content", ""), d.get("prompt_eval_count", 0),
                         d.get("eval_count", 0), time.time() - t0, raw=d)

@register_backend("openai_compat")
class OpenAICompatClient(LLMClient):
    """Any OpenAI-style /chat/completions endpoint (OpenRouter, Groq, ...). Falls back json_schema -> json_object on HTTP 400."""
    def __init__(self, model, base_url, api_key_env=None, api_key=None, temperature=0.0, max_tokens=1000,
                 seed=None, timeout_s=60, structured_mode="json_schema", extra_body=None,
                 extra_headers=None, max_tokens_param="max_tokens"):
        self.model, self.base_url, self.timeout = model, base_url.rstrip("/"), timeout_s
        self.api_key, self.api_key_env = api_key, api_key_env
        self.temperature, self.max_tokens, self.seed = temperature, max_tokens, seed
        self.mode, self.extra, self.headers = structured_mode, extra_body or {}, extra_headers or {}
        self.max_tokens_param = max_tokens_param; self.mode_fallbacks = 0

    def _key(self):
        k = self.api_key or (os.environ.get(self.api_key_env) if self.api_key_env else None)
        if not k: raise LLMError(f"missing API key (env var {self.api_key_env})")
        return k

    def _payload(self, system, user, schema, max_tokens=None):
        if self.mode == "json_schema":
            rf = {"type": "json_schema", "json_schema": {"name": "action", "strict": True, "schema": schema}}
        else:
            rf = {"type": "json_object"}
            system += "\nReply with ONE JSON object matching this JSON Schema:\n" + json.dumps(schema)
        p = {"model": self.model, "temperature": self.temperature, self.max_tokens_param: max_tokens or self.max_tokens, 
             "response_format": rf, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], **self.extra}
        if self.seed is not None: p["seed"] = self.seed
        return p

    def complete(self, system, user, schema, num_ctx=None, max_tokens=None):
        headers = {"Authorization": f"Bearer {self._key()}", "Content-Type": "application/json", **self.headers}
        t0 = time.time()
        try:
            d = _post(f"{self.base_url}/chat/completions", headers, self._payload(system, user, schema, max_tokens), self.timeout)
        except LLMHTTPError as e:
            if e.status in (400, 404) and self.mode == "json_schema":
                self.mode = "json_object"; self.mode_fallbacks += 1
                d = _post(f"{self.base_url}/chat/completions", headers, self._payload(system, user, schema, max_tokens), self.timeout)
            else: raise
        try: text = d["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError): raise LLMError(f"unexpected response shape: {str(d)[:200]}")
        u = d.get("usage") or {}
        return LLMResult(text, u.get("prompt_tokens", 0), u.get("completion_tokens", 0), time.time() - t0, raw=d)

@register_backend("mock")
class MockClient(LLMClient):
    """Offline backend for tests/pipeline checks. `responder(system, user) -> str`."""
    def __init__(self, responder=None, **_):
        self.responder = responder or (lambda s, u: json.dumps(
            {"battery_mode": "IDLE", "ev_mode": "DEADLINE_AWARE", "applied_rule_ids": [], "decision_summary": "mock"}))
        self.calls = []
    def complete(self, system, user, schema, num_ctx=None, max_tokens=None):
        self.calls.append((system, user)); return LLMResult(self.responder(system, user), 10, 10, 0.0)

def build_client(profile_cfg):
    cfg = dict(profile_cfg); t = cfg.pop("type")
    if t not in _BACKENDS: raise LLMError(f"unknown backend type {t!r}; registered: {sorted(_BACKENDS)}")
    return _BACKENDS[t](**cfg)

def load_client(path="config/model.json", profile=None):
    c = json.load(open(path)); name = profile or c["active_profile"]
    if name not in c["profiles"]:
        raise LLMError(f"unknown profile {name!r}; available: {sorted(c['profiles'])}")
    client = build_client(c["profiles"][name])
    log_path = os.environ.get("LLM_DEBUG_LOG") or c.get("debug_log")
    return LoggingClient(client, log_path, profile=name) if log_path else client

'''
Enable logging either way:

run:
LLM_DEBUG_LOG=runs/debug/llm_calls.jsonl python run.py --controllers llm --profile ollama_qwen3-8b ...

or add: "debug_log": "runs/debug/llm_calls.jsonl" to model.json. Leaving both unset disables logging.
'''
class LoggingClient(LLMClient):
    """Wraps any client; appends one JSON line per call (including failures) to `path`."""
    def __init__(self, inner, path, profile="unknown", log_schema=False):
        self.inner, self.path, self.profile, self.log_schema = inner, path, profile, log_schema
        self.context, self._n = {}, 0
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def __getattr__(self, name):          # expose inner attributes (model, mode, etc.)
        return getattr(self.inner, name)

    def set_context(self, **kw):          # episode_id, hour, attempt, etc.
        self.context = kw

    def complete(self, system, user, schema, num_ctx=None, max_tokens=None):
        self._n += 1
        rec = dict(seq=self._n, ts=time.strftime("%Y-%m-%dT%H:%M:%S"), profile=self.profile,
                   model=getattr(self.inner, "model", None), **self.context, req_num_ctx=num_ctx, req_max_tokens=max_tokens, system=system, user=user)
        if self.log_schema:
            rec["schema"] = schema
        t0 = time.time()
        try:
            res = self.inner.complete(system, user, schema, num_ctx, max_tokens)
        except Exception as e:
            rec.update(error=f"{type(e).__name__}: {e}", latency_s=round(time.time() - t0, 3))
            self._write(rec)
            raise
        rec.update(response_text=res.text, prompt_tokens=res.prompt_tokens,
                   completion_tokens=res.completion_tokens, latency_s=round(res.latency_s, 3), raw=res.raw)
        self._write(rec)
        return res

    def _write(self, rec):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")