# jev-tier-router

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) plugin that picks the
model for each turn. When you run on the Claude Subscription DirectSDK provider, it asks
[TypeSafe Jev](https://openrouter.ai/typesafe/jev-1.13) (through OpenRouter's Decisions
API) once per user turn whether a small fast model can handle the request. Depending on
the answer, the outgoing model becomes `claude-haiku-5-5[1m]` (simple) or
`claude-opus-5-5[1m]` (complex).

> Status: experimental snapshot, published as-is. It was built and tested in Oct 2026
> against Hermes >= 0.21.5. The author ran it briefly and then disabled it, so treat it
> as a working starting point, not a maintained product.

## Behaviour

- **One decision per turn.** Every request in an agent loop uses the same model, so the
  prompt-cache prefix stays stable. The decision is cached on the turn's user text.
- **Fail-open everywhere.** A missing key, a network error, a 2 s timeout or a malformed
  answer means no rewrite.
- **Only clear calls change the model.** A score of 0.85 or above goes to Haiku and 0.35
  or below goes to Opus. Anything in between keeps your session's model.
- **Only acts on its own provider.** Requests to provider
  `claude-subscription-directsdk-experimental` whose model is one of the two rungs. Every
  other request passes through byte-identical.
- **Thinking and effort settings are left alone.**
- **Holdback.** By default 10% of would-be downgrades deterministically stay on Opus, so
  you can measure what the cheap tier would have cost. Set `JEV_HOLDBACK_PCT=0` to
  turn it off.
- **Logging.** Every decision goes to `$HERMES_HOME/logs/jev-tier-router-decisions.jsonl`.
  Entries hold the event, models, score and turn hash, never your text.

**Disclosure:** each turn's text (truncated to 4,000 characters) is sent to TypeSafe via
OpenRouter. That costs about $0.000013 per call.

## Install

```bash
hermes plugins install vhsgreed/jev-tier-router
hermes plugins enable jev-tier-router
```

It uses the same `OPENROUTER_API_KEY` as your OpenRouter provider (from the environment
or `$HERMES_HOME/.env`).

## Test

```bash
pip install pytest && python -m pytest -q tests     # 13 tests, no network
```

## License

MIT (see `LICENSE`).
