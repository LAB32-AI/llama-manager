# Backlog

Loosely-prioritized list of known issues and improvements. Not a roadmap —
items land independently when someone picks them up.

## Bugs

- [ ] **`/restart` doesn't re-spawn the instance.** `POST /api/instances/{name}/restart`
  stops the running `llama-server` child but leaves the instance in state
  `stopped` (no error, `restarts: null`); an explicit `POST .../start` brings it
  back fine. Suspected cause in `manager.go` `RestartInstance`: the stop path
  cancels the per-instance supervisor context, and the follow-up start either
  races the teardown or the supervisor is never re-armed. Observed on the MI50
  rig (2026-05-23) when changing `context_length` and restarting GLM.

## Agentic-coding / agentic-task tuning

Context: 4×MI50 (32 GB each, gfx906), GLM-4.5-Air-Q4_K_M, used as an
OpenAI-compatible backend for agentic coding clients (opencode, aider).
Build verified working: llama.cpp `5306f4b` (2026-05-21). `--jinja` is on and
both single- and multi-turn tool calling work; f16/f16 KV cache is correct
(do NOT quantize KV — it degrades tool-calling accuracy).

The manager currently hardcodes the `llama-server` arg list. None of the flags
below are reachable. **Enabler:** add an `extra_args` config field (global
default + per-instance override) that appends arbitrary flags to the launch
command, then apply and benchmark the flags below.

- [ ] **Add `extra_args` (global + per-instance)** to config, arg builder, the
  Settings/instance UI, and `example.config.yaml`. Unlocks everything below
  without hardcoding each flag. (`server_bin` stays locked per the trust model;
  these are only flags to the already-trusted binary.)
- [ ] **`-np 1`** for single-developer use. With no `-np`, llama.cpp auto-creates
  4 slots that share one KV pool. A lone request can still use the full 32K, but
  consecutive turns of one agent conversation may land on different slots and
  lose the cached prefix → the whole prompt is reprocessed each turn. One slot
  guarantees prefix reuse across the agent loop. (Use more slots only for
  genuinely concurrent agents/users, and raise `-c` accordingly.)
- [ ] **`-ub 2048 -b 2048`** (physical/logical batch). Biggest prompt-processing
  win for agentic flows, where every turn ships a large system prompt + tool
  schemas + file context. Reported ~8s→3s on a 4K prompt. Costs some compute-
  buffer VRAM (we have ~55 GB free at 32K).
- [ ] **`--cache-reuse 256`** — reuse KV across turns via KV shifting for stable
  prefixes. Pairs with `-np 1`. (Multi-turn test currently shows `cached_tokens: 0`,
  i.e. no reuse today.)
- [ ] **`-fa on`** — flash attention. Defaults to `auto`; explicitly enabling it
  cuts KV memory and speeds attention, but mainline FA on gfx906 can be slow —
  benchmark on/off and keep whatever is faster. (A `llama.cpp-gfx906` fork has
  MI50-tuned FA kernels if mainline underperforms.)
- [ ] **`--reasoning-format` / `--reasoning-budget`** — GLM-4.5 is a thinking
  model; responses already split `reasoning_content` from `content` (good for
  opencode). Consider exposing these to cap thinking tokens for latency-sensitive
  agentic steps.
- [ ] **Context**: 32K is comfortable (KV ≈ +2.8 GB over 16K; ~55 GB free).
  64K trivial; the model's native max is 131072 and still fits (~+17 GB).

## Notes / refs

- llama.cpp function-calling: https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md
- Offline agentic coding with llama-server: https://github.com/ggml-org/llama.cpp/discussions/14758
- opencode providers (OpenAI-compatible): https://opencode.ai/docs/providers/
- gfx906/MI50 builds & FA kernels: https://github.com/iacopPBK/llama.cpp-gfx906
