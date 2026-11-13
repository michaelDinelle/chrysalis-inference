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

README.md shows what Chrysalis is. This file is how to change it.

## Layout

- engine/api.h declares the C API every engine implements, the chrysalis_* functions chrysalis/engine.py calls;
  engine/gguf.h, sampler.h and lookup.h are the host code every backend shares: the GGUF reader, the sampler that
  matches llama.cpp's, and prompt lookup.
- engine/<backend>/ (hip for the RX 7900, cuda for NVIDIA, metal for Apple Silicon) has a build.mk and one engine per
  model, 27b/ and flash-next/, each a shared library that includes engine/api.h. Every backend reads the same model files.
- A HIP engine has at most three streams of its own. ROCm gives a process four hardware queues and hipBLASLt's setup
  keeps one on the default stream for good, and streams take queues as they get work: with a fourth, two of them share
  a queue in some processes and run one after the other (Flash-Next's 4K prompt took 2.19 s instead of 1.72 s in about
  one process in six, until its expert loader moved onto the copy stream). `AMD_LOG_LEVEL=3` logs each queue a stream
  takes ("Selected queue", with its refCount); a fifth queue costs the KV cache VRAM and decode about 2%.
- chrysalis/models.py picks the backend from the platform (CHRYSALIS_BACKEND overrides it), with each backend's own
  settings in ON; bench/measure.py does the same for the bench.
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
vast.ai was destroyed on 2026-11-11: its verdicts stay in the ledger, and it judges nothing more. The hook (`git config core.hooksPath .githooks`) holds every commit to that.

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
  pictured), `make build/test_cpujob_fn && build/test_cpujob_fn`, `make test_voice` (the voice service: its features against
  transformers', turns, echo and barge-in over its WebSocket), `make test_web` (the web UI's sampler against
  the engine's, its server against a fake Chrysalis and a fake OpenCode, the page in headless Chromium, and Firefox past and trusting the network's
  certificate; certutil makes their trust: the files side by side, as many as the free memory holds, sharing one
  Chromium). `cd web && bun tests.ts ./<file>.test.ts` runs some of them.
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

## The RX 7900's PCIe link

The 7900 XTX's slot link (root port 00:01.1 to the card's own switch at 01:00.0) comes up at 16, 8 or 2.5 GT/s depending
on the boot: over 15 boots with the same BIOS (ASUS TUF B650-PLUS WIFI 3886, AGESA ComboAm5PI 1.3.0.1b), 8 came up at
16, 3 at 8 and 4 at 2.5, never at 5, whatever the kernel, the off time or the display. At 2.5 GT/s DMA runs at 3.6 GB/s
instead of ~25, and Flash-Next writes and loads slower. Don't blame the link for a number, and don't answer it with a
reseat, BIOS advice or a self-restart: the user wants it fixed for good, on any machine, at full speed.

What was measured as root on 2026-10-30 (passwordless sudo, `setpci`, amdgpu debugfs):
- It is not a speed cap. LnkCtl2's target is 16 GT/s on both ends, and Linux 7.1+ lifts a firmware 2.5 GT/s cap by
  itself at boot (commit 72780f796468) and logs "removing 2.5GT/s downstream link speed restriction"; a slow boot logs
  only the kernel's "limited by 2.5 GT/s" at the first PCI scan, so the speed is set before Linux runs.
- It is the board's power-on equalization. Speeds above 5 GT/s need a tuning round per speed (8, then 16). On a 2.5
  boot LnkSta2 reads `EqualizationComplete+ EqualizationPhase1-` on both ends ("complete" only means the attempt ended;
  phase 1 failed) and Phy16Sta is all minus (16 was never tried). An 8 boot is the 16 round failing. Check phase 1, not
  "complete": LnkSta2 bit 2 for 8 GT/s, Phy16Sta (ECAP 0x26 +0x0c) bit 1 for 16.
- After a failed power-on round the card caps itself. Its link controller's PCIE_LC_SPEED_CNTL (amdgpu debugfs
  `amdgpu_regs_pcie` at byte address (NBIF base 5, 0x04040000 from ip_discovery, + 0x28900a4) * 4 = 0x1a340290; fields in
  nbio_4_3_0_sh_mask.h as PSWUSP0_PCIE_LC_SPEED_CNTL) read 0x07e02a07 on the 2.5 boot (target-speed override on, at
  Gen3; advertising Gen3; "other side supports Gen4" clear) and 0x0fe03b67 on a 16 boot (override at Gen4).
- Nothing retries it. The card's power management drops the link to 2.5 GT/s x1 when idle and back to x16 under load;
  it raises the speed only when both rounds passed at power-on. The driver's own mode1 reset (`amdgpu_gpu_recover`)
  doesn't retrain the slot link, and forcing `pp_dpm_pcie` is accepted and ignored.
- Retuning from the root port works while the card is busy at x16: target 8 GT/s (`CAP_EXP+30.w=3:f`), Perform
  Equalization (`ECAP19+4.w=1:1`), Retrain Link (`CAP_EXP+10.w=20:20`). Each round is a coin flip, as at power-on;
  retrying until LnkSta2 reads 001f reached 8.0 GT/s x16. With the card capped at Gen3, a 16 target never starts a
  round. Retraining while the card idles at x1 does nothing.
- False leads: the Lane Error Status bits (they refill within seconds at 2.5 x1 from width switching), the root ports'
  equalization presets (P4 everywhere on both boots), the card's failed-attempt counters (clear).
- The fix is system/chrysalis-pcie-link, run as root by system/chrysalis-pcie-link.service at boot, after every wake,
  and (system/90-chrysalis-pcie-link.rules) whenever a GPU's driver comes up; `chrysalis.sh` offers to install it once
  on Linux with an AMD GPU of 4 GB or more, and `chrysalis.sh link` installs it later. For each GPU link running below
  what both ends support whose 8 or 16 GT/s round failed, it lifts the card's own cap (amdgpu with NBIF 4.3), then
  retrains at 8 GT/s and at 16 with Perform Equalization until each round passes (12 tries each), and puts the link
  back at 2.5 if the card stops answering. It does nothing on a good boot. Its pieces were each checked on this
  machine (the 8 GT/s retune on a 2.5 boot; the cap blocking 16 and lifting it restoring 16 on a 16 boot; retraining
  from idle), but not yet end to end on a 2.5 boot: check `journalctl -u chrysalis-pcie-link -b` after the next one.
