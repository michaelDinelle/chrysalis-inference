# Chrysalis Inference: notes for coding agents

README.md shows what Chrysalis is. This file is how to change it.

## Layout

- engine/api.h declares the C API every engine implements, the chrysalis_* functions chrysalis/engine.py calls;
  engine/gguf.h, sampler.h and lookup.h are the host code every backend shares: the GGUF reader, the sampler that
  matches llama.cpp's, and prompt lookup.
- engine/<backend>/ (hip for the RX 7900, metal for Apple Silicon) has a build.mk and one engine per model, 27b/ and
  flash-next/, each a shared library that includes engine/api.h. Every backend reads the same model files.
- chrysalis/models.py picks the backend from the platform (CHRYSALIS_BACKEND overrides it), with each backend's own
  settings in ON; bench/measure.py does the same for the bench.
- tests/<backend>/ hold kernel checks; tests/*.py check any backend through the C API.
- A new backend: engine/<name>/build.mk building build/libchrysalis and build/libchrysalis_fn, with `kernels` and
  `kernels_fn`, its model checks in BACKEND_TESTS and BACKEND_TESTS_FN (the builds they need in TEST_BUILDS and
  TEST_BUILDS_FN), and LLAMA_INC, LLAMA, LLAMA_LIBS and CXX_HOST for the llama.cpp references; its name in
  chrysalis/models.py and bench/measure.py; its kernel checks in tests/<name>/; and in bench/limits.py, how to read its
  GPU's memory bandwidth and FP16 rate, which the bench's share of the GPU's limit divides by.

## Bench

`make bench` measures both models and the hooks (`git config core.hooksPath bench/hooks`) hold every commit to its
parent on every machine bench/results.json has numbers from: a change to a backend's engine needs its numbers from a
machine running that backend. Measure there, then `data/venv/bin/python bench/bench.py merge <its bench/results.json>`
here before committing.

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
  needs bit-identical repeats within one process pins them (`chrysalis_tune` "split", "gfrac", "experts"), as
  tests/test_snapshot.py does. Fresh processes given the same requests repeat each other to the bit: its CPU, DMA, disk
  and prompt-split calibration is kept in data/cache/flash-next-calibration.txt, and pool moves land at fixed steps.
- Tests check, they don't time: a timing is not a check.

## Running them

- CPU only: `make build/test_sampler && build/test_sampler` (the sampler against llama.cpp's own samplers),
  `make build/llama_params && data/venv/bin/python tests/test_requests.py` (requests against llama.cpp's server's parser),
  `data/venv/bin/python tests/test_cache.py`,
  `data/venv/bin/python tests/test_lanes.py` (requests in flight, sub-agents beside a session's title),
  `data/venv/bin/python tests/test_chat.py`, `data/venv/bin/python tests/test_documents.py` (which PDF pages are
  pictured), `make build/test_cpujob_fn && build/test_cpujob_fn`, `make test_voice` (the voice service: its features against
  transformers', turns, echo and barge-in over its WebSocket), `make test_web` (the web UI's sampler against
  the engine's, its server against a fake Chrysalis, the page in headless Chromium, and Firefox past and trusting the network's
  certificate; certutil makes their trust).
- With Chrysalis running Flash-Next: `make test_chaos` (the chat for real, used oddly: replies through reloads, closed
  tabs and chat switches, several at once, Stop / regenerate / edit mid-reply, tools, settings, a 3D page, memory taken
  by another program; it restarts Chrysalis and switches models, so ask first, as for stopping it).
- GPU, with Chrysalis's model server stopped (the chat's Stop, which frees the GPU; the user runs Chrysalis: check the GPU
  is idle, ask once per task, then just say when it stops and for how long): `make test` (test_web, the 27B kernels,
  then the model in-process: chat format, cache, snapshots and generation (held answers' masked steps too), the server
  (its Anthropic and Responses APIs through those SDKs, which data/venv has; held answers and strict tool calls,
  thinking budgets, PDFs, videos, a long prompt read beside a reply), its next-token distributions against llama.cpp's,
  and pictures and videos: preprocessing, positions and the vision tower against transformers, prompts with pictures
  against llama.cpp; the transformers references need data/ref-venv: CPU torch, torchvision, transformers, Pillow, safetensors), `make test_fn` (Flash-Next
  kernels) and `make test_fn_model` (the Flash-Next model in-process: generation and snapshots, the server, llama.cpp's
  logits, pictures and videos).
- On a Mac with Apple Silicon and 32 GB or more, the same targets build and check the Metal engines: `make kernels`,
  `make kernels_fn`, `make test` and `make test_fn_model`. Their references come from data/llama.cpp at the commit Linux
  pins, built with Metal (`cmake -B build && cmake --build build --target llama ggml ggml-base`), and the PDF checks
  need poppler (`brew install poppler`). The Metal engines have no vision tower yet, so their tests leave pictures out.
- On the RX 7900, Flash-Next takes ~47 GiB of system RAM, in Chrysalis or in-process, and more as its expert pool moves (until 4 GB is left, given back below 3 GB). Check MemAvailable before loading it, and run it
  in its own scope (`systemd-run --user --scope ...`): systemd-oomd kills whole terminal scopes under memory pressure.
