# Chrysalis Inference: notes for coding agents

README.md shows what Chrysalis is. This file is how to change it.

## Comments

None: no comments or docstrings anywhere in the repo, in any language, including code inside strings.

## Tests

- Check observable behavior against something independent of the code under test: a scalar CPU reference, ggml's
  quantizers and dequantizers, llama.cpp's logits, the official chat template, or bit-identity with another path.
  Don't add tests that mirror the implementation, restate constants, or only exercise a fake.
- Every check must be able to fail. One that can be skipped fails when it is skipped (a path that got no work proves
  nothing), and a case is named for what it exercises.
- Prove a new or fixed check: break the product code on purpose, watch the check fail, revert.
- Flash-Next's CPU / GPU split and its expert pool move with history and change results by rounding: a check that
  needs bit-identical repeats pins them (`chrysalis_tune` "split", "gfrac", "experts"), as tests/test_snapshot.py does.
- Tests check, they don't time: a timing is not a check.

## Running them

- CPU only: `make build/test_sampler && build/test_sampler`, `data/venv/bin/python tests/test_cache.py`,
  `data/venv/bin/python tests/test_lanes.py` (requests in flight, sub-agents beside a session's title),
  `data/venv/bin/python tests/test_chat.py`, `data/venv/bin/python tests/test_documents.py` (which PDF pages are
  pictured), `make build/test_cpujob_fn && build/test_cpujob_fn`, `make test_voice` (the voice service: its features against
  transformers', turns, echo and barge-in over its WebSocket), `make test_web` (the web UI's sampler against
  the engine's, its server against a fake Chrysalis, the page in headless Chromium, and Firefox past and trusting the network's
  certificate; certutil makes their trust).
- With Chrysalis running Flash-Next: `make test_chaos` (the chat for real, used oddly: replies through reloads, closed
  tabs and chat switches, several at once, Stop / regenerate / edit mid-reply, tools, settings, a 3D page, memory taken
  by another program; it restarts Chrysalis and switches models, so ask first, as for stopping it).
- GPU, with Chrysalis's model server stopped (`systemctl --user stop chrysalis`; the user runs Chrysalis: check the GPU
  is idle, ask once per task, then just say when it stops and for how long): `make test` (test_web, the 27B kernels,
  then the model in-process: chat format, cache, snapshots and generation (held answers' masked steps too), the server
  (its Anthropic and Responses APIs through those SDKs, which data/venv has; held answers and strict tool calls,
  thinking budgets, PDFs, videos, a long prompt read beside a reply), its next-token distributions against llama.cpp's,
  and pictures and videos: preprocessing, positions and the vision tower against transformers, prompts with pictures
  against llama.cpp; the transformers references need data/ref-venv: CPU torch, torchvision, transformers, Pillow, safetensors), `make test_fn` (Flash-Next
  kernels) and `make test_fn_model` (the Flash-Next model in-process: generation and snapshots, the server, llama.cpp's
  logits, pictures and videos).
- Flash-Next takes ~47 GiB of system RAM, in Chrysalis or in-process, and more as its expert pool moves (until 4 GB is left). Check MemAvailable before loading it, and run it
  in its own scope (`systemd-run --user --scope ...`): systemd-oomd kills whole terminal scopes under memory pressure.
