# kiln

An inference engine written for exactly one model on exactly one GPU: **Qwen3.8-27B** on a
**Radeon RX 7900 XTX** (gfx1100, 24 GB). It serves an OpenAI-compatible API built for coding
agents such as OpenCode: full 262K context, prompt caching that makes every agent turn an
append, and MTP speculative decoding. A second engine runs **Qwen3.8-Flash-Next** (a 48-layer,
512-expert MoE) split between the GPU and system RAM; requests pick the model and kiln switches
(see [Qwen3.8-Flash-Next](#qwen38-flash-next)).

About 3,400 lines: C++/HIP kernels and model code in `engine/` (2,400), the Python server in `kiln/` (1,000).

## Use it

```sh
./kiln.sh install        # once: systemd user services + ~/.local/bin/kiln (does not enable autostart)

kiln start               # loads the model (~1.5 s), starts the web chat, prints both URLs:
                         #   API http://127.0.0.1:8080/v1, web chat http://127.0.0.1:8081
kiln start --lan         # the same, also on the local network: prints its URLs and a QR code for phones
kiln stop                # stops both and frees the GPU (also: restart, status)
kiln logs                # one line per request: prompt/cached tokens, TTFT, tok/s, draft acceptance
```

`--lan` is plain HTTP for a trusted home network, without logins; the firewall has to let other devices in
(TCP 8080-8081), and a plain `kiln start` or `kiln restart` goes back to this machine only. Either way both servers
answer only requests addressed to an IP address or to this machine's name, which keeps web pages from other sites
out (DNS rebinding). `systemctl --user enable kiln kiln-web` starts both at login. `kiln serve` runs the server in the
foreground instead of as a service (it takes `--port`, `--draft`, `--use`, ...: `kiln serve -h`). Models:
`qwen3.8-27b` (thinking, reasoning effort `xhigh` by default; OpenCode variants pick `medium`/`low`)
and `qwen3.8-27b-fast` (no thinking; also used for session titles), and the same pair for
`qwen3.8-flash-next`. One model is on the GPU at a time: a request naming the other one switches
(titles never do); `kiln status` says which is loaded. OpenCode reaches it as an OpenAI-compatible
provider at `http://127.0.0.1:8080/v1`, set up in `~/.config/opencode/opencode.json`.

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
| Decode at 131K of real code (thinking, MTP) | **80 tok/s** (37 without drafts) | ~17 tok/s at 128K (reported) |
| Decode at 260K of real code | **64-69 tok/s** (31 without drafts) | |
| Prefill, 2K-token chunks | **1,350 tok/s** | 1,015 tok/s |
| Prefill, short appends (one forward pass) | **16 tokens in 43 ms, 64 in 73 ms, 256 in 222 ms** | |
| Prefill at 128K | **~740 tok/s**; filling 260K from empty: 377 s (690 tok/s) | ~390 tok/s (reported) |
| 1,024-token append at 260K | **2.4 s** to the first token | |
| Agent turn TTFT (small tool output) | **0.11–0.18 s**; large outputs add ~1,300 tok/s | full re-prefill whenever history diverges |
| OpenCode compaction | **0.2 s**: served from the cached conversation | full re-read: minutes at 200K |
| New OpenCode session TTFT (warm prefix) | **0.08 s** same project, **0.27 s** another project or day | several seconds |
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
here, 18% smaller: what makes 262K fit). Decode and verify attention: a workgroup of 8 waves per key split
and KV head walks 128-key stages; each wave scores 16 of the keys for all query rows (K rows straight from
memory into WMMA fragments), the waves agree on the softmax max through LDS, and each accumulates the
output for its own 32 of the 256 dims, so a wave holds 16 accumulator registers instead of 128 and every KV
byte is read once for all draft tokens. K and V decode to fp16 with packed arithmetic (bytes into the
mantissa of 1024 by `v_perm`, two 5-bit values per and-or). 0.67 ms per layer at 262K (~700 GB/s; the read
floor is 0.50 ms), 0.79 ms for a 4-token verify. Prefill attention: 16-wave workgroups share 32-key LDS
stages, and each stage's raw K/V is loaded into registers during the previous stage's softmax and P·V, so
DRAM latency hides behind WMMA work: ~80 TFLOPS at 260K.

**Prefill** — chunks of up to 384 tokens (agent turns) run a fused GEMM that dequantizes weights in
registers straight into WMMA fragments: the weights are read once at 4–6 bits instead of being expanded
to fp16 first (that copy costs ~140 ms per pass). WMMA wants each fragment in both half-waves; here each
half-wave dequantizes or loads half of it and they swap with `v_permlanex16` (the halves may hold K in
different orders as long as A and B agree), halving that work. Longer chunks use hipBLASLt fp16 GEMMs
(~95 TFLOPS on this card vs ~60 for rocBLAS) on weights dequantized per layer. Token-parallel conv,
LDS-staged recurrent scan (16 tokens per stage, so every workgroup of a layer is resident at once, and
4-way split dot products for a short dependency chain per token).

**Caching (TTFT)** — `kiln/chat.py` renders the chat template straight to token ids, byte-identical
to the official template, and replays assistant turns kiln generated as their exact ids (looked up by
their text, tool calls and reasoning, so two turns that said the same "OK" stay apart), so each agent step
only prefills the new tool output. Snapshots (recurrent state in RAM, KV pages shared copy-on-write) at
the end of the tool block, of the static system prompt (cut before OpenCode's date and environment
lines, so it is shared across projects and days), of the system turn, and of every prompt: that last one
keeps the next token's distribution, so a resent prompt (a retry, a redraw) needs no prefill at all and an
agent step is one forward pass. A prompt is always cut at these points, so a seed draws the same reply
from any cached state. Pinned buffers for all snapshot slots are made during warm-up (Flash-Next: one 1.9 GB
region whose pages fault in while the warm-up runs; made one at a time, each cost a request 0.1-0.4 s in the
minutes after a load). They live in RAM and VRAM only: nothing is written to disk, so a restart starts cold.
Tokenization is memoized per message, so re-rendering a 200K-token conversation each turn costs ~2 ms.
Side requests (OpenCode's session title, sent 8 ms before the session's first request, and its summaries)
go last: they start after 0.3 s without an agent request, and their instructions are kept like the tool
block. OpenCode specifics: the date in its environment block is frozen per conversation, plan-mode
reminders are kept where they first appeared, and its compaction request (normally a full re-read of the
older part of the conversation) is answered by appending the summary request to the cached conversation,
rendered with that conversation's own settings. Extra system messages from plugins are merged into the
first one. Special-token text inside files stays text. `KILN_TRACE=1` adds each request's time split
(cache lookup, prefill pieces, snapshots, first step) to its log line; `tools/bench_opencode.py` replays
OpenCode 2.0.25's real requests (`tools/opencode2.json`) against a running server.

## Qwen3.8-Flash-Next

48 layers (36 gated delta net, 12 sparse "QSA" attention), hidden size 2560 widened to 4 hyper-connection
streams, 512 experts per layer (10 routed + a shared one), a 54 GB n-gram embedding table read at layer 1,
and an MTP block with its own 512 experts. kiln runs one layout, sized so the model fits next to a desktop
on 64 GB of RAM with no swapping:

- **GPU (20.2 GB):** dense weights Q6_K / Q8_0, the LM head Q6_K, the MTP block (its experts Q3_K), and the
  5,600 experts used most (IQ4_XS gate/up, IQ4_NL down). 3.3 GB of KV covers the 262K context.
- **System RAM (42.6 GB):** the other 18,976 experts: Q3_K gate/up, the least used fifth of all experts Q2_K,
  IQ4_NL down. K-quants run at the CPU's full memory bandwidth (the IQ3 formats are compute-bound there).
  Flash-Next takes ~40 GiB of RAM in all.
- **Disk:** the n-gram table stays in the source file, memory-mapped; a token's 16 rows are requested
  together so the cold ones are read in parallel, and the next token's while the current step finishes.

Which experts sit in VRAM comes from routing statistics, blended: what the model picks while decoding in
OpenCode-style agent sessions (0.6; counted on the GPU with `KILN_FN_DECODE_COUNTS=file`, 69K tokens of
`tools/agent_sim.py` sessions over a dozen coding tasks), what it picks while reading code (0.25,
`tools_fn/route_stats.py`) and Unsloth's imatrix counts (0.15). Decode routing correlates little with
either: tuned on code prefill alone, the VRAM tier caught 41-45% of decode picks; this one catches ~59% on
held-out sessions (cross-validated by task), so a third fewer experts come from RAM per token, and 59% of
code prefill's (64% before).

**Decode.** Per layer the GPU routes (one fused router + top-10 kernel) and hands the cold picks over
through pinned coherent memory: the CPU workers (one per physical core) compute most of them with AVX-512
VNNI row kernels (each 64-weight register decoded straight from the GGUF blocks, group scales as lane vectors,
one horizontal sum per row: ~52-56 GB/s from RAM, near the ~62 GB/s the CPU's fabric link allows). The shared expert, the hot experts and the GPU's own share of the cold experts (read from
RAM over PCIe) run side by side in two launches (gate/up, then down), so the PCIe-bound picks hide the hot
ones. One kernel waits for the CPU, sums and updates the residual streams. While the CPU works, 8
workgroups on a second stream read the next layer's dense weights into the 96 MB Infinity Cache, where the
GPU reads them ~2x faster (more workgroups would crowd out the expert kernels). Wave reductions and the
router's top-10 use DPP / `v_permlanex16` lane exchanges and branch-free compares instead of LDS
round trips. CPU and PCIe share the RAM's bandwidth, which bounds the cold experts' part of a token. That
bandwidth idles while the GPU runs a layer's token mixer, so a fourth stream runs the layer's own MoE mixer
and router on a copy of its input residual meanwhile and publishes the predicted CPU picks (they catch ~70%
of the cold picks); the workers load those experts' weights into their caches until the real handoff
arrives (prefetch hints are dropped at this volume; plain loads are not). It changes no result and gains
1-2% at short context, ~4% at 64K.

**Prefill.** A prompt goes through in passes of 8,192 tokens. Within a pass, each layer runs its mixers,
attention and recurrent state over sub-chunks of 1,024 tokens, then its routed experts once for the whole
pass, so every expert in RAM is read once per pass rather than once per chunk. Hot experts run as grouped
WMMA GEMMs over work items of up to 256 of an expert's tokens: a workgroup dequantizes its 32 weight rows into
LDS once per half super-block and each of its waves runs 16 tokens against them (gate/up 8 waves, down 16:
up to ~80 TFLOPS). Each sub-chunk's picks and MoE inputs go to the host on the copy stream while the next
sub-chunks run, so planning a layer's experts waits only for the last one. The sparse attention gives
each (token, KV head) a workgroup of 8 waves over 128-key stages of the token's selected blocks, split by
output dims as in the 27B's decode kernel. Of the cold experts, the CPU computes the ones with few tokens in place (an
AVX-512 VNNI kernel with tokens as lanes) and the GPU DMAs the rest into a staging ring. The split follows a
cost model with rates measured at load and a feedback term that keeps the CPU and GPU finishing together. The
feedback runs for the first ten passes after a load and then stays put: the CPU and GPU round differently, so a
split that kept following timing made the same prompt prefill to different logits each time; the settled one
is as fast (frozen at any point between 1.5x and 2x the CPU estimate, within 2% of the adapting split, and up to
10% faster than its oscillation on long passes) and Flash-Next's results no longer depend on history.

**Memory.** The model is sized to fit, so there is nothing to guard: kiln loads it and runs. Switching to
the 27B frees all of it (no experts are kept in RAM), and switching back reloads in ~18 s. Pages of the
model file are dropped from the page cache once copied.

**Measured** (7800X3D, 64 GB DDR5, RX 7900 XTX):

| | |
|---|---|
| decode, no drafts / 2 MTP drafts | 66-68 / 82-85 tok/s (in a coding conversation: 62 / 84) |
| prefill 16K tokens | 6.2 s |
| append to a 16K conversation: 64 / 256 / 1,024 / 4,096 tokens | 0.18 / 0.42 / 0.82 / 1.89 s to the first token |
| at 260K context: decode, no drafts / 2 drafts | 57.7 / 65 tok/s (27B: 30.6 / 69 with 3 drafts) |
| at 260K context: 1,024-token append | 0.91 s to the first token (27B: 2.38 s) |
| filling 260K tokens from empty | 106 s, 2,450 tok/s (27B: 377 s) |
| load / switch from the 27B | 18 s / 19 s |
| OpenCode (`tools/bench_opencode.py`): first request after a load / new session / new day or project | 5.9 / 0.14 / 0.5 s to the first token |

Speed hardly changes with depth (`tools/bench_depth.py`, same code base as context for both models): only 12 of
the 48 layers attend, each to a fixed 2,048 selected cells plus the indexer's scan, where the 27B's full-attention
layers read the whole cache for every token.

**Quality.** Mean KL divergence against the Q8_0 original on llama.cpp (`tools_fn/compare_fn.py`): 0.011-0.016,
top-1 agreement 96-99%. A long text that is mostly code pasted into a user turn scores ~0.06: the model's
predictions inside user turns are chaotic (two kiln runs that differ only in rounding disagree by 0.025 there).

## Layout

```
engine/   common.h kernels.h gguf.h      constants, kernel entry points, GGUF reader
          load.cpp relayout.hip          loading: GGUF -> VRAM pipeline, KV pool, warm-up
          quant.hip                      weight arrays, GEMVs, dequantization
          attn.hip gdn.hip ops.hip       attention, gated delta net, norms / top-k
          model.cpp model.h api.cpp      weights, sequences, snapshots, forward, MTP, C API
kiln/     engine.py chat.py cache.py server.py __main__.py
web/      server.ts app.ts index.html style.css   Bun web UI
tools/    tests (make test: test_quant, test_attn, test_chat, test_snapshot, test_server; the last two take
          --use qwen3.8-flash-next), compare_ref.py, benchmarks (bench_opencode.py: OpenCode's real requests), agent_sim.py
engine_fn/                               the Flash-Next engine (same C API): cpu_moe.cpp cpu_vnni.cpp for the
                                         experts in RAM, moe.hip routing, ple.hip n-gram block, ...
tools_fn/ fn_quantize.py (builds kiln-fn.gguf), route_stats.py (routing counts for it), test_*_fn.cpp (make test_fn),
          compare_fn.py, bench_fn.py, bench_dec_fn.cpp / bench_moeg_fn.cpp / bench_cpudec_fn.cpp (decode / prefill MoE
          kernels / the CPU's decode handoff alone)
data/     (not code) rocm/ venv/ models/ llama.cpp/ (quantization tools + reference only)
```

## Rebuilding

`make` builds `build/libkiln.so` against the ROCm 10.1 tarball in `data/rocm`. The model file was
made from bartowski's Q8_0 with llama.cpp's quantizer:
`llama-quantize --allow-requantize --imatrix Qwen3.8-27B-imatrix.gguf --tensor-type-file tools/quant-mix.txt
--output-tensor-type q6_k --token-embedding-type iq4_xs Qwen3.8-27B-Q8_0.gguf kiln-mix.gguf IQ4_XS`.

`make fn` builds `build/libkiln_fn.so`. Its model file comes from Unsloth's Q8_0 split (plus the MTP file and
imatrix) via `python tools_fn/fn_quantize.py SRC_DIR data/models/flash-next/kiln-fn.gguf --hot 5600 --q2 0.2
--route data/models/flash-next/route_code.npy:0.25 --route data/models/flash-next/route_decode.npy:0.6` (~22 min);
`tools_fn/route_stats.py` measured the code-prefill counts on local code, `KILN_FN_DECODE_COUNTS` the decode counts
in `tools/agent_sim.py` sessions (`--model=qwen3.8-flash-next`).

## Notes

- One request runs at a time (agent loops are sequential); three sequences stay resident on the GPU,
  more are kept as snapshots in RAM.
- The GPU stays awake while the server runs (no wake-up latency) and suspends when it stops.
  Stop the server before suspending the machine.
- Images in prompts are replaced by a note: the vision tower is not loaded.
