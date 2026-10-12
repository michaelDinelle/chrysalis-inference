# kiln

An inference engine written for exactly one model on exactly one GPU: **Qwen3.8-27B** on a
**Radeon RX 7900 XTX** (gfx1100, 24 GB). It serves an OpenAI-compatible API built for coding
agents such as OpenCode: full 262K context, prompt caching that makes every agent turn an
append, and MTP speculative decoding.

About 3,400 lines: C++/HIP kernels and model code in `engine/` (2,400), the Python server in `kiln/` (1,000).

## Use it

```sh
./kiln.sh install        # once: systemd user services + ~/.local/bin/kiln (does not enable autostart)

kiln start               # loads the model (~1.5 s), starts the web chat, prints both URLs:
                         #   API http://127.0.0.1:8080/v1, web chat http://127.0.0.1:8081
kiln stop                # stops both and frees the GPU (also: restart, status)
kiln logs                # one line per request: prompt/cached tokens, TTFT, tok/s, draft acceptance
```

`systemctl --user enable kiln kiln-web` starts both at login. `kiln serve` runs the server in the
foreground instead of as a service (it takes `--port`, `--draft`, ...: `kiln serve -h`). Models:
`qwen3.8-27b` (thinking, reasoning effort `xhigh` by default; OpenCode variants pick `medium`/`low`)
and `qwen3.8-27b-fast` (no thinking; also used for session titles). OpenCode reaches it as an
OpenAI-compatible provider at `http://127.0.0.1:8080/v1`, set up in `~/.config/opencode/opencode.json`.

## Web UI

`web/` is a small Bun app: `server.ts` serves the page and proxies kiln's API on the same origin; `app.ts` is
plain TypeScript (marked, highlight.js, DOMPurify). Conversations live in the tab's sessionStorage: a reload
keeps them, closing the tab ends them, and nothing is stored on the server.
Replies stream with their reasoning (collapsible), each one shows time to first token, tok/s, how much of the
prompt came from cache and its seed. The Think switch picks the reasoning effort (Max, Medium, Low) or the
no-thinking model (Off). Settings holds the system prompt, the sampling values (temperature, top-p, top-k,
min-p, presence penalty, max reply tokens) and the seed; they apply to every conversation in the tab and are
kept with them. A blank field uses Qwen's recommended value for the thinking mode, a blank seed a new one for
every reply: set a reply's seed to draw it again. Esc stops a reply, the last turn can be regenerated or
edited, and the sidebar starts or unloads the model (sending a message starts it too). Assistant turns go back
to kiln exactly as received, so every turn of a conversation is a cache hit.
Development: `cd web && bun run dev` (hot reload), `bun run check`.

## Measured on this machine

| | kiln | llama.cpp (same GGUF, ROCm 10.1) |
|---|---|---|
| Context | **262,144** tokens (KV pool holds 276K) | 262K needs 3-bit weights or 4-bit KV |
| Decode, coding prompt, thinking, Qwen sampling | **100–114 tok/s** (MTP, 3 drafts) | 39.8 tok/s |
| Decode, greedy | **112–116 tok/s** | |
| Decode at 112K of real code (thinking, MTP) | **58 tok/s** (31 without drafts) | ~17 tok/s at 128K (reported) |
| Decode at 250K (synthetic context) | **57 tok/s** (23 without drafts) | |
| Prefill, 2K-token chunks | **1,350 tok/s** | 1,015 tok/s |
| Prefill, short appends (one forward pass) | **16 tokens in 43 ms, 64 in 73 ms, 256 in 222 ms** | |
| Prefill at 128K | **~700 tok/s** (0→112K averages 877) | ~390 tok/s (reported) |
| Agent turn TTFT (small tool output) | **0.11–0.18 s**; large outputs add ~1,300 tok/s | full re-prefill whenever history diverges |
| OpenCode compaction | **0.2 s**: served from the cached conversation | full re-read: minutes at 200K |
| New session TTFT (warm prefix) | **0.24 s** | several seconds |
| Load: start to serving, model file in the page cache | **1.5 s**, first reply at 1.8 s | |
| Load from disk (file not cached) | **3.7–3.9 s**, first reply at 4.3–4.5 s | |
| Unload: stop to process gone / VRAM free | **0.04 s / 0.19 s** | |

Quality:
- Kernels: mean KL divergence vs llama.cpp on identical weights 0.002 (98–100% top-1 agreement),
  far below quantization noise (`tools/compare_ref.py`).
- Weights: mean KL vs the Q8_0 original on 40K tokens of code and docs is 0.0100 (97.99% same top
  token), against 0.0111 (97.74%) for llama.cpp's own IQ4_XS recipe with the same imatrix (`tools/quality.sh`).

## How it gets there

**Weights** — `tools/quant-mix.txt`: IQ4_XS for the bulk (FFN, projections), Q6_K for the sensitive
`ssm_out` / `attn_output` / output head, Q8_0 for K/V projections and the MTP head; 15.9 GB. At load
the GGUF blocks are split into aligned arrays (same bits) so kernels use 16-byte loads. Token
embeddings stay in pinned host RAM and are read by the GPU over PCIe (saves 0.7 GB of VRAM).

**Loading** (`engine/load.cpp`) — from a stopped server to serving in ~1.5 s, overlapping everything that
can overlap: the GPU wakes from runtime suspend (~0.6 s, started when the process starts) while Python
starts, the tokenizer loads and a thread pulls any uncached part of the file into the page cache (the disk
streams ~4.9 GB/s). Writer threads then copy the tensors in file order straight into a VRAM staging ring
through the large BAR (non-temporal stores, ~21 GB/s, one pass over host RAM) and the GPU relayouts each
32 MB piece into place as it lands. VRAM is allocated meanwhile in ~1 GB pieces (an allocation holds a HIP
lock while the kernel clears it); the KV pool takes what is left, the staging ring included once the
weights are in. The server listens from the start (`/health` answers 503 until the model is in) and then
warms up under its GPU lock (~0.3 s): pinned host memory for the embeddings (relayouted on the CPU) and
the snapshot slots, and the first kernel launches, which load the code objects. Both stall other HIP calls,
so they stay out of the weight upload.

**Decode** — int8 dot-product GEMVs (activations quantized per 32 values) at ~800–920 GB/s; up to three
matrices per launch (q|k|v, qkv|z); SwiGLU fused with quantization of the down projection's input; one
fused kernel per recurrent layer (gates, conv, L2 norms, delta-rule scan with the 128×128 state in
registers, gated norm). MTP drafts on the GPU without host round trips, then one verify pass;
rejection sampling against greedy drafts keeps sampling exact; the recurrent state is committed by
replaying the accepted tokens. Each reply seeds its sampler from the request's `seed` (or a random one) and the
response carries it back (`seed`, in the finish_reason chunk when streaming): the same prompt, settings and
seed give the same tokens, from a cold cache, a snapshot or the live conversation. A prompt read in different
chunks (an older conversation re-read after a restart) has slightly different logits and may drift after a
while. The presence penalty counts the current reply's tokens only.

**Attention** — paged KV cache (64-token pages), K int8, V 5-bit Hadamard-rotated (as accurate as int8
here, 18% smaller: what makes 262K fit). Decode attention is LDS-free WMMA: K rows, Q and transposed V
rows go straight into WMMA fragments, V scales folded into the probabilities, splits sized to fill the
GPU. Prefill attention: 16-wave workgroups sharing 32-key LDS stages, ~60 TFLOPS at 128K.

**Prefill** — chunks of up to 384 tokens (agent turns) run a fused GEMM that dequantizes weights in
registers straight into WMMA fragments: the weights are read once at 4–6 bits instead of being expanded
to fp16 first (that copy costs ~140 ms per pass). WMMA wants each fragment in both half-waves; here each
half-wave dequantizes or loads half of it and they swap with `v_permlanex16` (the halves may hold K in
different orders as long as A and B agree), halving that work. Longer chunks use hipBLASLt fp16 GEMMs
(~95 TFLOPS on this card vs ~60 for rocBLAS) on weights dequantized per layer. Token-parallel conv,
LDS-staged recurrent scan.

**Caching (TTFT)** — `kiln/chat.py` renders the chat template straight to token ids, byte-identical
to the official template, and replays assistant turns kiln generated as their exact ids, so each agent
step only prefills the new tool output. Snapshots (recurrent state in RAM, KV pages shared
copy-on-write) at the end of the tool block, the static system prompt, the system turn and every prompt.
They live in RAM and VRAM only: nothing is written to disk, so a restart starts cold. Tokenization is memoized
per message, so re-rendering a 200K-token conversation each turn costs ~2 ms. OpenCode specifics:
the date in its environment block is frozen per conversation, plan-mode reminders are kept where
they first appeared, and its compaction request (normally a full re-read of the older part of the
conversation) is answered by appending the summary request to the cached conversation, rendered with
that conversation's own settings. Extra system messages from plugins are merged into the first one.
Special-token text inside files stays text.

## Layout

```
engine/   common.h kernels.h gguf.h      constants, kernel entry points, GGUF reader
          load.cpp relayout.hip          loading: GGUF -> VRAM pipeline, KV pool, warm-up
          quant.hip                      weight arrays, GEMVs, dequantization
          attn.hip gdn.hip ops.hip       attention, gated delta net, norms / top-k
          model.cpp model.h api.cpp      weights, sequences, snapshots, forward, MTP, C API
kiln/     engine.py chat.py cache.py server.py __main__.py
web/      server.ts app.ts index.html style.css   Bun web UI
tools/    tests (make test: test_quant, test_attn, test_chat, test_snapshot, test_server), compare_ref.py,
          benchmarks, agent_sim.py
data/     (not code) rocm/ venv/ models/ llama.cpp/ (quantization tools + reference only)
```

## Rebuilding

`make` builds `build/libkiln.so` against the ROCm 10.1 tarball in `data/rocm`. The model file was
made from bartowski's Q8_0 with llama.cpp's quantizer:
`llama-quantize --allow-requantize --imatrix Qwen3.8-27B-imatrix.gguf --tensor-type-file tools/quant-mix.txt
--output-tensor-type q6_k --token-embedding-type iq4_xs Qwen3.8-27B-Q8_0.gguf kiln-mix.gguf IQ4_XS`.

## Notes

- One request runs at a time (agent loops are sequential); three sequences stay resident on the GPU,
  more are kept as snapshots in RAM.
- The GPU stays awake while the server runs (no wake-up latency) and suspends when it stops.
  Stop the server before suspending the machine.
- Images in prompts are replaced by a note: the vision tower is not loaded.
