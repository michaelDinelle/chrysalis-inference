# Chrysalis Inference: notes for coding agents

**Using the Hill Climb Anything skill is mandatory, at all times, for every change:
https://github.com/michaelDinelle/hill-climb-anything** (installed at ~/.claude/skills/hill-climb-anything). Load it
before touching anything and climb by it: choose a target, profile it, change one thing, gate it with `.hill/hill gate`
on every machine with a record, and report each verdict's table. It is not optional, and no task is too small or too
urgent for it.

**No guessed time limits and no fixed sleeps.** Never wrap a job in an arbitrary `timeout`, and never wait with a fixed
`sleep`: both either cut real work short or burn wall clock doing nothing. Run a long job in the background and act when
it completes; when something must wait, wait on the condition itself (the GPU reads empty, a file exists, a process has
exited).

**There are no trade-offs.** Giving up some of one reward (quality, speed, memory, context, library size, lines of code)
for another is not a move here. Don't weigh one, don't offer one as an option, and don't ask which reward matters more:
asking is proposing. A loss is a problem to solve, never a price. No second model file, lighter build, mode or variant
for a hard machine: one file and one path that win everywhere. Don't state ceilings ("it can't do better than X on this
card"): say what was measured, what the largest remaining cost is and what changes next. When Strata or llama.cpp gets
a number Chrysalis doesn't on the same hardware, that number is reachable: read how they do it and get it at
Chrysalis's quality. When one machine wins and another loses with the same file, compare their measured cost per unit
of work before explaining anything. Delete as you climb: a second copy of a file, or a new file an existing one could
have covered, is a loss. Don't stop to report or ask mid-climb; report at a verdict.

README.md shows what Chrysalis is. This file is how to change it.

## Layout

- engine/api.h declares the C API every engine implements, the chrysalis_* functions chrysalis/engine.py calls;
  engine/gguf.h, sampler.h and lookup.h are the host code every backend shares: the GGUF reader, the sampler that
  matches llama.cpp's, and prompt lookup. engine/cpu_exact.cpp is the x86 CPU expert kernels of the AMD and NVIDIA
  Flash-Next engines: cpu_exact_isa.h written once, compiled for AVX-512 VNNI, AVX-VNNI, AVX-512 and AVX2 with the
  same bits, the fastest the CPU has picked at load (exact_use_path forces one).
- engine/os.h is everything the AMD and NVIDIA engines ask of the operating system (files, memory, threads, how much
  VRAM the system grants: Windows' budget, and on Linux what other programs hold on the card), for Linux, macOS and
  Windows; engine/cpu_exact.{h,cpp} and cpu_exact_isa.h are Flash-Next's exact CPU expert code, one source compiled for
  AVX-512 VNNI, AVX-VNNI, AVX-512 and AVX2 and picked when the library loads.
- On Windows the HIP SDK's compiler is older than Linux's: a result exchanged between lanes (lane_xor, a ballot) must be
  used by every lane, or it can be moved into a branch only some lanes run (the 27B wrote NaN after 130 tokens from
  this; tests/hip/test_attn.hip checks each row's maximum). A stream that only the GPU waits on is submitted with
  `submit` after its work is queued, and nothing passes a C file descriptor into an engine: Windows takes handles.
- engine/<backend>/ (hip for the RX 7900, cuda for NVIDIA, metal for Apple Silicon) has a build.mk and one engine per
  model, 27b/ and flash-next/, each a shared library that includes engine/api.h. Every backend reads the same model files.
- A HIP engine has at most three streams of its own. ROCm gives a process four hardware queues and hipBLASLt's setup
  keeps one on the default stream for good, and streams take queues as they get work: with a fourth, two of them share
  a queue in some processes and run one after the other (Flash-Next's 4K prompt took 2.19 s instead of 1.72 s in about
  one process in six, until its expert loader moved onto the copy stream). `AMD_LOG_LEVEL=3` logs each queue a stream
  takes ("Selected queue", with its refCount); a fifth queue costs the KV cache VRAM and decode about 2%.
- chrysalis/models.py picks the backend from the platform (CHRYSALIS_BACKEND overrides it), with each backend's own
  settings in ON; bench/measure.py does the same for the bench.
- chrysalis.sh sets a machine up (Python, Bun, the CUDA and Metal builds) and hands everything else to web/launch.ts, the
  launcher written once for every system (Chrysalis.exe on Windows is that file compiled; windows/build.ps1 and
  windows/package.ps1 build the engines and the folder there), which hands its terminal to web/cli.ts, which runs the model server and web/server.ts,
  shows each request as it finishes and takes commands: web/cli-core.ts is the servers, the state and the commands,
  web/cli-ui.tsx draws it with Ink (the switches, the status line, the prompt), and without a terminal it is plain
  lines. The web server tells it who asks and what to ask over Bun's IPC. It shows the servers' own output only after `verbose`, and keeps
  nothing: no log, no history file.
- An incognito chat is a kind of chat, not a mode, and works the same on every system. web/harness.ts runs a second
  OpenCode for incognito threads (`hidden`, host id "incognito") with its database in memory (OPENCODE_DB=:memory:) and
  routes each thread to the OpenCode that holds it. Every request of an incognito chat reaches the model server marked
  X-Chrysalis-Unkept, so chrysalis/cache.py keeps that request's prompt snapshots (the tools and the system prompt,
  tokens and all) in RAM only and saves no expert pool while one runs. The incognito OpenCode's other files go in a
  folder of its own on a drive that is only ever memory (`ramRoot` in web/platform.ts: /dev/shm on Linux, a RAM disk
  mounted with hdiutil on macOS, WinFsp's in-memory file system on Windows, build/ramdrive.exe, installed at the first
  incognito chat), gone when the chats end. Nothing of an incognito chat may be written anywhere else. The
  page marks such a chat `incognito`, lists it nowhere, keeps it and its files out of the browser's storage, and
  deletes it the moment it is left (`leave` in web/app.ts). Anything new that writes a chat to disk has to be closed
  there too.
- The context ring in the message box opens OpenCode's own numbers for a thread and its context breakdown
  (web/breakdown.ts is OpenCode's estimate, ported; the notice is web/THIRD-PARTY.md), which web/harness.ts runs over
  the messages OpenCode's `/api/session/:id/context` returns. The page adds nothing OpenCode does not hold. Every
  control of the message box keeps its place in every chat: one that cannot act is dimmed, not hidden.
- tests/<backend>/ hold kernel checks; tests/*.py check any backend through the C API.
- A new backend: engine/<name>/build.mk building build/libchrysalis and build/libchrysalis_fn, with `kernels` and
  `kernels_fn`, its model checks in BACKEND_TESTS and BACKEND_TESTS_FN (the builds they need in TEST_BUILDS and
  TEST_BUILDS_FN), and LLAMA_INC, LLAMA, LLAMA_LIBS and CXX_HOST for the llama.cpp references; its name in
  chrysalis/models.py and bench/measure.py; and its kernel checks in tests/<name>/.

## Rewards

Every reward in hill.json is held by `.hill/hill`. A change is promoted only when no reward is worse on any machine with
a record and at least one is better: no trade-offs, no net gains, no "within noise" after the fact, no blaming the
machine. `.hill/hill gate` is the only judge; if it is wrong, fix it and gate again. The ledger is append-only. Rewards
are never dropped, re-keyed or loosened. Only the user can make an exception, and agents never propose one. Harness
changes (bench/, hill.json, .hill/) land on their own, never with a product change. A change is the hill's to judge when it
touches hill.json's measured paths (engine/, chrysalis/, the Makefile) outside its not_measured ones (chrysalis/voice,
which the server never imports): web/, tests/ (the test targets are tests/tests.mk), the voice service and these notes
commit with no gate. The README has no Rewards table and
`hill table` isn't run: report each verdict's table to the user instead, and keep the README's charts and its For agents
numbers true to what Chrysalis does now. media/charts.json holds those numbers, and `data/venv/bin/python media/charts.py`
is the only thing that writes the charts and the tables, both from that file: never edit them by hand. `python3
media/charts.py --check` fails if either differs from it, and the hook and `make test_web` run it.

A gate measures the working tree against HEAD in the same session, both models on the GPU with Chrysalis's model server
stopped; bench/hill_ready.py refuses to start while anything else holds the GPU or free memory is short. Measuring time
is a reward too: a gate takes minutes, deep context is restored from snapshots primed once per machine, and the bench
only gets faster. Every machine with a record (the RX 7900 here and the M4 Max on studio) judges each change: run
`.hill/hill gate` there and append its new .hill/ledger.jsonl lines here before committing. The RTX 5070 rented on
vast.ai was destroyed on 2026-11-11: its verdicts stay in the ledger, and it judges nothing more. The RTX 3090 rented
there (with a Xeon W-2133) holds a record from 2026-11-15 and judges every change to engine/cuda/ for as long as it is
rented. The hook (`git config core.hooksPath .githooks`) holds every commit to that.

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
  and prompt-split calibration, and on CUDA the RAM its mirror may use, is kept in data/cache/flash-next-calibration.txt,
  and pool moves land at fixed steps.
- Tests check, they don't time: a timing is not a check.

## Running them

- CPU only: `make build/test_sampler && build/test_sampler` (the sampler against llama.cpp's own samplers),
  `make build/llama_params && data/venv/bin/python tests/test_requests.py` (requests against llama.cpp's server's parser),
  `data/venv/bin/python tests/test_cache.py`,
  `data/venv/bin/python tests/test_lanes.py` (requests in flight, sub-agents beside a session's title),
  `data/venv/bin/python tests/test_chat.py`, `data/venv/bin/python tests/test_documents.py` (which PDF pages are
  pictured, and what pdfium reads against what poppler read), `data/venv/bin/python tests/test_video.py` (the frames picked
  from videos and their pixels, against what ffmpeg gave), `make build/test_cpujob_fn && build/test_cpujob_fn`,
  `make build/test_cpu_exact && build/test_cpu_exact` (every CPU expert path this CPU has against a scalar reference and
  each other, to the bit), `make test_voice` (the voice service: its features against transformers', turns, echo and
  barge-in over its WebSocket), `make test_web` (the web UI's sampler against
  the engine's, its server against a fake Chrysalis and a fake OpenCode, the terminal against a fake model server, incognito against a fake OpenCode binary with a store of its own, the page in headless Chromium, and Firefox past and trusting the network's
  certificate; certutil makes their trust: the files side by side, as many as the free memory holds, sharing one
  Chromium). `cd web && bun tests.ts ./<file>.test.ts` runs some of them.
- GPU, with Chrysalis's model server stopped (`model unload` in its terminal frees the GPU; the user runs Chrysalis: check the GPU
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
  pins, built with Metal (`cmake -B build && cmake --build build --target llama ggml ggml-base`). Flash-Next's prompts
  with a picture are checked against llama.cpp on its Q8_0 source, which a Mac does not hold: tests/test_vision.py leaves
  llm.tokens.i32, llm.pos.i32 and llm.embd.f32 in build/vision/fn; run build/ref_logits_mm_fn on them where the source
  is, into the file the test names.
- On an NVIDIA GPU, the same targets build and check the CUDA engines (nvcc from /usr/local/cuda, the native architecture
  unless ARCH is set): `make kernels`, `make kernels_fn`, `make test` and `make test_fn_model`, with data/llama.cpp built
  for CUDA (`cmake -B build -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON && cmake --build build`). On a card too small for a
  model, the 27B keeps the layers that don't fit in pinned RAM (streamed to VRAM for prompts, read across PCIe for
  writing) and Flash-Next starts the VRAM experts that don't fit in its RAM mirror; CHRYSALIS_CTX and CHRYSALIS_FN_CTX
  set the context they keep VRAM for. How far Flash-Next's mirror may grow is kept on the machine's calibration line: the
  first process to load it there keeps the RAM the engine may hold (what was free when it began loading plus what that
  process already held, a container's own limit as the bound, less 5 GB) and what the engine holds beyond its experts, so
  fresh processes repeat each other to the bit whatever is free (tests/test_mirror_budget.py); below 250 MB free it still
  gives memory back. The vision checks on any machine read the first safetensors shard of Qwen's own release
  (model-00001 of Qwen/Qwen3.8-27B and of Qwen/Qwen3.8-Flash-Next) as data/models/<model>/<model>-shard1.safetensors.
  Their llama.cpp references are kept in build/vision by their inputs: on a card too small for the 27B, make them with
  build/ref_logits_mm on a machine that holds it on its GPU (llama.cpp on the CPU is further from that than the checks
  allow), as Flash-Next's come from a machine with its Q8_0 source.
- On the RX 7900, Flash-Next takes ~47 GiB of system RAM, in Chrysalis or in-process, and more as its expert pool moves (until 4 GB is left, given back below 3 GB). Check MemAvailable before loading it, and run it
  in its own scope (`systemd-run --user --scope ...`): systemd-oomd kills whole terminal scopes under memory pressure.
