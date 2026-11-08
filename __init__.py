"""jev-tier-router: per-turn complexity routing between subscription rungs.

Rewrites the outgoing model of Claude Subscription DirectSDK requests between
claude-haiku-5-5[1m] and claude-opus-5-5[1m] based on one Jev classification per
user turn (TypeSafe Jev via OpenRouter's Decisions API, existing OPENROUTER_API_KEY).

Invariants:
- Acts only on provider `claude-subscription-directsdk-experimental` requests
  whose model is one of the two rungs; anything else is byte-identical.
- One decision per user turn (sticky cache keyed on the turn's user text), so
  every agent-loop request in a turn uses the same model and the CLI prompt
  cache prefix stays stable. Consecutive turns with equal text reuse the
  decision too.
- Fail-open everywhere: no key, network error, timeout, or malformed payload
  means no rewrite. Mid-confidence scores mean no rewrite (session default wins).
- Thinking/effort is never touched (Haiku 5.5 / Opus 5.5 reject thinking-off).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request

TARGET_PROVIDER = "claude-subscription-directsdk-experimental"
OPUS = "claude-opus-5-5[1m]"
HAIKU = "claude-haiku-5-5[1m]"
RUNGS = {"claude-opus-5-5": OPUS, "claude-haiku-5-5": HAIKU}
ALIASES = {"opus": "claude-opus-5-5", "haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5"}
SIMPLE_THRESHOLD = 0.85
COMPLEX_THRESHOLD = 0.35
JEV_URL = "https://openrouter.ai/api/alpha/decisions"
JEV_MODEL = "~typesafe/jev-latest"
TIMEOUT_S = 2.0
MAX_TURN_CHARS = 4000
MAX_CACHED_TURNS = 512

STATE_PREFIX = "A coding assistant received this user request:\n"
QUESTION = {
    "simple_enough": {
        "type": "noul",
        "instructions": (
            "A small fast model could answer this user request correctly "
            "without deep multi-step reasoning"
        ),
    }
}

# turn key -> (score or None, chosen model or None)
_decisions: dict = {}


def _hermes_home() -> str:
    return os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")


def _api_key() -> str:
    key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if key:
        return key
    candidates = (
        os.path.join(_hermes_home(), ".env"),
        os.path.expanduser("~/.secrets/OPENROUTER_API_KEY"),
    )
    for path in candidates:
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip("\"'")
            if os.path.basename(path) == "OPENROUTER_API_KEY" and line:
                return line
    return ""


def _message_text(message) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
    return ""


def _last_user_text(messages) -> str:
    for message in reversed(list(messages or [])):
        if isinstance(message, dict) and message.get("role") == "user":
            text = _message_text(message)
            if text.strip():
                return text
    return ""


def _turn_key(session_id: str, user_text: str) -> str:
    digest = hashlib.sha256(user_text.encode("utf-8", "replace")).hexdigest()[:16]
    return f"{session_id or '-'}|{digest}"


def _remember(key: str, value: tuple) -> None:
    _decisions[key] = value
    while len(_decisions) > MAX_CACHED_TURNS:
        _decisions.pop(next(iter(_decisions)))


def _log(entry: dict) -> None:
    try:
        path = os.path.join(_hermes_home(), "logs", "jev-tier-router-decisions.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **entry}
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _ask_jev(user_text: str) -> float:
    """Return the noul score (1.0 = small model is enough). Raises on any failure."""
    key = _api_key()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")
    body = json.dumps(
        {"model": JEV_MODEL, "state": STATE_PREFIX + user_text, "questions": QUESTION}
    ).encode()
    req = urllib.request.Request(
        JEV_URL,
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        payload = json.loads(resp.read().decode())
    return float(payload["answers"]["simple_enough"]["noul"])


def _rung(model: str):
    """(normalized base, canonical request model) or (None, None) when not a routed rung."""
    base = model.removesuffix("[1m]")
    base = ALIASES.get(base, base)
    if base not in RUNGS:
        return None, None
    return base, RUNGS[base]


HOLDBACK_PCT = int(os.environ.get("JEV_HOLDBACK_PCT", "10"))  # registered default


def _holdback(turn_key: str) -> bool:
    """Deterministic holdback, reproducible from the turn key alone.

    Off when HOLDBACK_PCT is 0 (the test suite sets JEV_HOLDBACK_PCT=0 because
    behavior tests define the router contract; holdback has its own tests)."""
    if HOLDBACK_PCT <= 0:
        return False
    return int(hashlib.sha256(turn_key.encode()).hexdigest(), 16) % 100 < HOLDBACK_PCT


def on_llm_request(**kwargs):
    request = dict(kwargs.get("request") or {})
    if kwargs.get("provider") != TARGET_PROVIDER:
        return {"request": request}
    model = request.get("model")
    if not isinstance(model, str) or not model:
        return {"request": request}
    base, current = _rung(model)
    if base is None:
        return {"request": request}
    user_text = _last_user_text(request.get("messages") or request.get("input") or [])
    if not user_text.strip():
        return {"request": request}

    key = _turn_key(str(kwargs.get("session_id") or ""), user_text)
    if key in _decisions:
        score, chosen = _decisions[key]
    else:
        try:
            score = _ask_jev(user_text[:MAX_TURN_CHARS])
            if score >= SIMPLE_THRESHOLD:
                chosen = HAIKU
            elif score <= COMPLEX_THRESHOLD:
                chosen = OPUS
            else:
                chosen = None
        except Exception as exc:  # fail-open: never rewrite on classifier trouble
            score, chosen = None, None
            _log({"event": "error", "detail": str(exc)[:200]})
        _remember(key, (score, chosen))

    if chosen and chosen != current:
        if chosen == HAIKU and _holdback(key):
            # research holdback (PLAN-routing.md phase B): 10% of downgrades stay
            # on the expensive model, deterministically, to measure what the
            # cheap tier would have cost in failures. Logged, never silent.
            _log({"event": "downgrade_holdback", "from": current, "score": score,
                  "turn": key, "holdback": True})
            return {"request": request}
        request["model"] = chosen
        action = "downgrade" if chosen == HAIKU else "upgrade"
        _log({"event": action, "from": current, "to": chosen, "score": score, "turn": key})
        return {
            "request": request,
            "source": "jev-tier-router",
            "reason": f"noul={score:.2f} -> {chosen}",
        }
    _log({"event": "keep", "model": current, "score": score, "turn": key})
    return {"request": request}


def on_llm_execution(**kwargs):
    """Wire-verification hook: logs the model at the provider-call boundary."""
    request = kwargs.get("request") or {}
    if kwargs.get("provider") == TARGET_PROVIDER:
        _log({"event": "exec_model", "model": request.get("model")})
    return kwargs["next_call"](request)


def register(ctx):
    ctx.register_middleware("llm_request", on_llm_request)
    ctx.register_middleware("llm_execution", on_llm_execution)
