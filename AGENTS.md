# kiln: notes for coding agents

README.md says what kiln is, how it is built and what it measures. This file is how to change it.

## Comments

Comments say why: a constraint, a hardware or library quirk, a measured reason, a data layout, an invariant the code
cannot show. Don't restate names or signatures, narrate steps, label sections, or record history ("was", "new",
"no longer", what an earlier version did). When a change makes a comment wrong, fix or delete it in the same change:
a stale comment steers people and agents wrong. `make lint` catches the mechanical cases (tools/lint_comments.py).

## Tests

- Check observable behavior against something independent of the code under test: a scalar CPU reference, ggml's
  quantizers and dequantizers, llama.cpp's logits, the official chat template, or bit-identity with another path.
  Don't add tests that mirror the implementation, restate constants, or only exercise a fake.
- Every check must be able to fail. One that can be skipped fails when it is skipped (a path that got no work proves
  nothing), and a case is named for what it exercises.
- Prove a new or fixed check: break the product code on purpose, watch the check fail, revert.
- Flash-Next's CPU / GPU split and its expert pool move with history and change results by rounding: a check that
  needs bit-identical repeats pins them (`kiln_tune` "split", "gfrac", "experts"), as tools/test_snapshot.py does.
- Tests check, benchmarks measure: timings belong in the bench_* tools.

## Running them

- CPU only: `make build/test_sampler && build/test_sampler`, `data/venv/bin/python tools/test_cache.py`,
  `data/venv/bin/python tools/test_chat.py`, `make build_fn/test_cpujob_fn && build_fn/test_cpujob_fn`, `make test_web`
  (the web UI's sampler against the engine's, its server against a fake kiln, the page in headless Chromium), `make lint`.
- GPU, with kiln's model server stopped (`systemctl --user stop kiln`; the user runs kiln: check the GPU is idle, ask
  once per task, then just say when it stops and for how long): `make test` (test_web, the 27B kernels, then the model
  in-process: chat format, cache, snapshots and generation (held answers' masked steps too), the server (its Anthropic
  and Responses APIs through those SDKs, which data/venv has; held answers and strict tool calls, thinking budgets, PDFs,
  videos, a long prompt read beside a reply), its next-token distributions against llama.cpp's, and pictures and videos:
  preprocessing, positions and the vision tower against transformers, prompts with pictures against llama.cpp; the
  transformers references need data/ref-venv, see README Rebuilding), `make test_fn` (Flash-Next kernels) and
  `make test_fn_model` (the Flash-Next model in-process: generation and snapshots, the server, llama.cpp's logits,
  pictures and videos).
- Flash-Next takes ~47 GiB of system RAM, in kiln or in-process. Check MemAvailable before loading it, and run it in
  its own scope (`systemd-run --user --scope ...`): systemd-oomd kills whole terminal scopes under memory pressure.
