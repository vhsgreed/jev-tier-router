"""Offline tests for jev-tier-router. Run: python3 tests/test_router.py (or pytest)."""
from __future__ import annotations

import importlib.util
import os
import sys

os.environ["JEV_HOLDBACK_PCT"] = "0"  # behavior tests run with holdback off

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = importlib.util.spec_from_file_location(
    "jev_tier_router", os.path.join(HERE, "..", "__init__.py")
)
router = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(router)
router._log = lambda entry: None  # keep test runs out of the real decision log

PROV = router.TARGET_PROVIDER


def make_request(model="claude-opus-5-5[1m]", user="hi"):
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": user},
        ],
    }


def call(request, provider=PROV, session_id="s1", asker=None):
    calls = []
    if asker is not None:
        def counted(text):
            calls.append(text)
            return asker(text)
        router._ask_jev = counted
    router._decisions.clear()
    out = router.on_llm_request(request=request, provider=provider, session_id=session_id)
    return out["request"], len(calls)


def test_other_provider_untouched():
    req, n = call(make_request(), provider="openrouter", asker=lambda t: 0.99)
    assert req["model"] == "claude-opus-5-5[1m]", req
    assert n == 0


def test_non_rung_model_untouched():
    req, n = call(make_request(model="claude-fable-5-1"), asker=lambda t: 0.99)
    assert req["model"] == "claude-fable-5-1"
    assert n == 0


def test_simple_downgrades_to_haiku():
    req, n = call(make_request(user="What is 2+2?"), asker=lambda t: 0.98)
    assert req["model"] == router.HAIKU, req
    assert n == 1


def test_complex_stays_opus():
    req, n = call(make_request(user="Design a distributed cache"), asker=lambda t: 0.2)
    assert req["model"] == "claude-opus-5-5[1m]", req
    assert n == 1


def test_mid_score_untouched():
    req, n = call(make_request(user="medium"), asker=lambda t: 0.5)
    assert req["model"] == "claude-opus-5-5[1m]", req


def test_fail_open_on_classifier_error():
    def boom(text):
        raise RuntimeError("network down")
    req, n = call(make_request(user="whatever"), asker=boom)
    assert req["model"] == "claude-opus-5-5[1m]", req


def test_sticky_within_turn():
    def counter(text):
        counter.n += 1
        return 0.98
    counter.n = 0
    router._ask_jev = counter
    router._decisions.clear()
    req1 = router.on_llm_request(request=make_request(user="hi"), provider=PROV, session_id="s")["request"]
    req2 = router.on_llm_request(request=make_request(user="hi"), provider=PROV, session_id="s")["request"]
    assert req1["model"] == req2["model"] == router.HAIKU
    assert counter.n == 1, counter.n


def test_upgrade_when_session_on_haiku():
    req, n = call(make_request(model="claude-haiku-5-5", user="hard problem"), asker=lambda t: 0.1)
    assert req["model"] == router.OPUS, req


def test_alias_model_names():
    req, n = call(make_request(model="opus[1m]", user="quick"), asker=lambda t: 0.99)
    assert req["model"] == router.HAIKU, req


def test_no_user_text_untouched():
    request = {"model": "claude-opus-5-5[1m]", "messages": [{"role": "assistant", "content": "x"}]}
    req, n = call(request, asker=lambda t: 0.99)
    assert req["model"] == "claude-opus-5-5[1m]"
    assert n == 0


def test_original_request_never_mutated():
    original = make_request(user="hi")
    snapshot = dict(original)
    call(original, asker=lambda t: 0.98)
    assert original == snapshot


def test_holdback_deterministic_and_bounded():
    router.HOLDBACK_PCT = 10
    keys = [f"turn-{i}" for i in range(2000)]
    first = [router._holdback(k) for k in keys]
    assert first == [router._holdback(k) for k in keys], "not deterministic"
    rate = sum(first) / len(first)
    assert 0.06 <= rate <= 0.14, f"holdback rate {rate:.3f} outside 10% band"
    router.HOLDBACK_PCT = 0


def test_holdback_keeps_expensive_model():
    router.HOLDBACK_PCT = 10
    held = [k for k in (f"turn-{i}" for i in range(500)) if router._holdback(k)]
    assert held, "need at least one held key to test"
    called = {}
    router._ask_jev = lambda text: 0.99  # always simple enough -> downgrade
    router._decisions.clear()
    key = held[0]
    user_text = "x" * 10
    router._turn_key = lambda sid, txt: key  # force the held key
    out = router.on_llm_request(provider=router.TARGET_PROVIDER,
                                request={"model": router.OPUS,
                                         "messages": [{"role": "user", "content": user_text}]},
                                session_id="s1")
    assert out["request"]["model"] == router.OPUS, out
    assert "reason" not in out
    router.HOLDBACK_PCT = 0


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    sys.exit(1 if failures else 0)
