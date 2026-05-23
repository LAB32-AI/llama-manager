# Backlog

Loosely-prioritized list of known issues and improvements. Not a roadmap —
items land independently when someone picks them up.

## Bugs

- [x] **`/restart` doesn't re-spawn the instance.** Fixed (commit 9c42ae0,
  deployed 2026-05-23, verified on the rig). Two teardown races killed the
  freshly-started process: (1) `StopInstance` cancelled the old supervisor's
  context without waiting for that goroutine to exit, so it ran `inst.Stop()`
  on the new process — fixed with a `done` channel + `detachSupervisor()` that
  blocks until the old goroutine exits; (2) `inst.Stop()` only sends `Kill()`
  and returns, so the old process's exit goroutine woke after the new `Start()`,
  saw `state != Stopped`, and marked it crashed — fixed by draining `<-exitCh`
  after `Stop()` in `runWithRestart`. Regression test: `TestRestartReSpawnsInstance`.

## Agentic-coding / agentic-task tuning

Context: 4×MI50 (32 GB each, gfx906), GLM-4.5-Air-Q4_K_M, used as an
OpenAI-compatible backend for agentic coding clients (opencode, aider).
Build verified working: llama.cpp `5306f4b` (2026-05-21). `--jinja` is on and
both single- and multi-turn tool calling work; f16/f16 KV cache is correct
(do NOT quantize KV — it degrades tool-calling accuracy).

- [x] **`extra_args` (global + per-instance)** — done (commit 704a26d). Appends
  arbitrary flags to the launch command; per-instance `*[]string` overrides the
  global. (`server_bin` stays locked per the trust model; these are only flags to
  the already-trusted binary.)
- [x] **Applied to GLM on the rig (2026-05-23):**
  `extra_args: [-np 1, -ub 2048, -b 2048, --cache-reuse 256]`.
  - **`-np 1`** — the real win. Without `-np`, llama.cpp auto-made 4 slots
    sharing one KV pool; consecutive turns of one agent conversation could land
    on different slots and lose the cached prefix. One slot dedicates the full
    32K and guarantees prefix reuse across the agent loop. `total_slots` 4→1.
    (Use more slots only for genuinely concurrent agents, and raise `-c`.)
  - **`-ub 2048 -b 2048`** — only ~4% prompt-processing gain on gfx906
    (334→348 tok/s on a 2.4K prompt; the card is compute-bound, not batch-
    overhead-bound). Kept anyway — costs ~6 GB compute buffer (50 GB still free)
    and may help more on the 10–20K prompts real agentic sessions produce.
  - **`--cache-reuse 256`** — reuse KV across turns via KV shifting for shifted
    prefixes (identical prefixes already cache 100%).
- [ ] **`-fa on`** — benchmarked on/off: **no measurable decode difference**
  (~32 tok/s both ways) on gfx906; mainline FA isn't tuned for this card. Left
  out. Revisit with the `llama.cpp-gfx906` fork (MI50-tuned FA kernels) if ever
  building a custom llama.cpp.
- [x] **`--reasoning-budget 512`** — applied to GLM (2026-05-23). GLM-4.5 is a
  thinking model (responses split `reasoning_content` from `content`). Unlimited
  reasoning was a real problem: on a routine coding prompt it spent the entire
  600-token budget *thinking* and never emitted the function. With a 512-token
  budget it caps the thinking and is forced to produce the answer. Tune via
  Settings → extra args (`-1` unlimited, `0` off, `N>0` budget; build supports
  arbitrary N, env `LLAMA_ARG_THINK_BUDGET`). Speed-vs-quality dial — lower for
  snappy simple steps, higher/unlimited for hard planning.
- [ ] **Context**: now at 32K (KV ≈ +2.8 GB over 16K). 64K trivial; native max
  131072 still fits (~+17 GB). ~50 GB free with the agentic flags applied.

## Notes / refs

- llama.cpp function-calling: https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md
- Offline agentic coding with llama-server: https://github.com/ggml-org/llama.cpp/discussions/14758
- opencode providers (OpenAI-compatible): https://opencode.ai/docs/providers/
- gfx906/MI50 builds & FA kernels: https://github.com/iacopPBK/llama.cpp-gfx906
