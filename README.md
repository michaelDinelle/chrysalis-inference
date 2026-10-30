<p align="center"><img src="media/butterfly.webp" width="300" alt="Chrysalis's butterfly in its sea of characters"></p>

<h1 align="center">Chrysalis</h1>

<p align="center">A local inference engine, API server and web chat for Qwen3.8-27B and Qwen3.8-Flash-Next on an AMD Radeon RX 7900 XTX or a Mac with Apple Silicon.</p>

<p align="center">
  <a href="LICENSE"><img alt="MIT license" src="https://img.shields.io/badge/license-MIT-e24d85"></a>
  <a href="https://x.com/michaeldinelle"><img alt="Follow on X" src="https://img.shields.io/badge/follow-%40michaeldinelle-000?logo=x"></a>
  <a href="https://ko-fi.com/michaeldinelle"><img alt="Support on Ko-fi" src="https://img.shields.io/badge/Ko--fi-support-ff5e5b?logo=ko-fi&logoColor=white"></a>
</p>

<p align="center"><img src="media/pagoda.webp" width="900" alt="A voxel pagoda garden that Qwen3.8-Flash-Next wrote in Chrysalis, running in the browser"><br>
<sub>A voxel pagoda garden that Qwen3.8-Flash-Next wrote from one prompt in about two minutes, in Chrysalis on an RX 7900 XTX.</sub></p>

## Chrysalis vs Strata

Qwen3.8-Flash-Next on one PC. Each engine runs at its best for this PC, and the same client times every request.

<p align="center"><img src="media/vs-strata.png" width="860" alt="Chrysalis against Strata IQ3_S and Q2_0: time to first token and writing speed at five prompt lengths, error against Q8_0, load time, memory, and each one's share of this GPU's limit"></p>

Strata's IQ3_S is the closest to Chrysalis in quality, and the best Strata's setup offers a 64 GB PC. Chrysalis beats it
on every measure here: first tokens 1.8-2.9x sooner, writing 1.2-1.3x faster, 1.7x closer to the full model and 1.6x
faster to load, in less memory. Strata's Q2_0 is its speed setting: it writes 1.0-1.2x faster than Chrysalis and needs
10 GB less memory, at 5.6 times the error, and it failed one of its own sanity checks in one of the two runs. Reading a
100K prompt, Chrysalis reaches 30% of this GPU's limit and Strata 17-18%.

<details><summary>How it was measured, and every number</summary>

RX 7900 XTX (24 GB), Ryzen 7 7800X3D, 64 GB DDR5, Linux, one engine at a time.

- **Chrysalis**: its own Flash-Next file, 3.8 bits per expert weight, 63 GB plus a 54 GB lookup table read from disk. The
  most used experts are IQ4_XS (gate and up) with IQ4_NL (down), the others Q3_K or Q2_K with IQ4_NL down, and attention
  and the remaining layers Q6_K and Q8_0.
- **Strata 0.2.6** (engine 0.1.50, its `linux-x86_64-hip-rx7900` build) with its RX 7900 switches on: the hipBLASLt
  tuning table, prompt attention on the matrix cores and the exact long-prompt scorer. Its own `strata tune auto` found
  nothing faster. IQ3_S is 3.3 bits (55 GB plus a 29 GB table) and Q2_0 2.3 bits (38 GB plus 29 GB).
- Short, 4K and 16K are Strata's standard benchmark, run through its own client code: its fixed prompts, Qwen's thinking
  defaults, 512-token answers and its 12 sanity checks. 100K is the same kind of records at 98,878 tokens, and the
  follow-up adds about 1,100 tokens to that chat. Each engine ran all of it twice: medians over 6 rounds, and over 10
  trials at 100K.
- KL is against Q8_0 over the same 1,082 tokens of prose, a technical explanation and Python. Chrysalis ran one token at
  a time, as it writes, with its experts moving as they do in use; Strata's files ran in llama.cpp.
- Memory is the peak private RAM of each server. File pages, which the kernel can drop at any time, are not counted.
- **Share of this GPU's limit**: the time the card's peak FP16 matrix rate needs for the prompt's math, or its memory
  bandwidth for the weights a pass reads, over the time taken. [bench/limits.py](bench/limits.py) reads both from the
  driver and the work from the model file. For Flash-Next it counts the whole model as if it were in VRAM, which it
  can't be on this card, and Strata doesn't report how many tokens each of its passes writes.

| | Chrysalis, 3.8-bit | Strata IQ3_S, 3.3-bit | Strata Q2_0, 2.3-bit |
| --- | ---: | ---: | ---: |
| First token, short prompt | **0.22 s** | 0.64 s | 0.41 s |
| First token, 4K prompt | **1.8 s** | 3.4 s | 3.0 s |
| First token, 16K prompt | **6.0 s** | 10.6 s | 9.8 s |
| First token, 100K prompt | **37.0 s** | 66.2 s | 60.7 s |
| First token, follow-up in the 100K chat | **1.1 s** | 1.9 s | 1.9 s |
| Writing, short prompt | 92.6 tok/s | 72.9 tok/s | **107.3 tok/s** |
| Writing, 4K prompt | 89.1 tok/s | 66.5 tok/s | **95.6 tok/s** |
| Writing, 16K prompt | 87.6 tok/s | 69.7 tok/s | **95.4 tok/s** |
| Writing, 100K prompt | 80.0 tok/s | 65.8 tok/s | **81.0 tok/s** |
| Writing, follow-up in the 100K chat | 82.3 tok/s | 69.7 tok/s | **97.4 tok/s** |
| Error vs Q8_0, mean KL over 1,082 tokens | **0.033** | 0.055 | 0.187 |
| Load time, to a first answer | **15.7 s** | 24.6 s | 17.4 s |
| Peak private RAM | 50.6 GB | 52.0 GB | **40.8 GB** |
| Share of this GPU's limit, reading 100K | **30%** | 17% | 18% |
| Share of this GPU's limit, writing at 100K | **22%** | not reported | not reported |

</details>

## Chrysalis vs llama.cpp, Qwen3.8-27B

The same file in Chrysalis and in llama.cpp, on the same GPU.

<p align="center"><img src="media/27b-vs-llamacpp.png" width="860" alt="Qwen3.8-27B in Chrysalis and in llama.cpp: time to first token and writing speed at five prompt lengths, load time, memory, and each one's share of this GPU's limit"></p>

Chrysalis reads prompts 1.4-2.6x sooner, writes 1.5-2.1x faster, starts 2.1x sooner and keeps 11 GB less in RAM. At
100K it reaches 62% of this GPU's limit reading the prompt and 55% writing, llama.cpp 40% and 27%.

<details><summary>How it was measured, and every number</summary>

- **The file**: Chrysalis's own 27B file, one file for both engines: 16 GB, entirely in VRAM, 4.65 bits per weight on
  average because each tensor has its own type, IQ4_XS for most and Q6_K or Q8_0 for the most sensitive.
- **llama.cpp** commit 9c2e0e4 (October 7, 2026) built for HIP, at its best on this card: every layer on the GPU, flash
  attention with an 8-bit KV cache (its 16-bit decode kernel for this model's 256-wide heads isn't built for this card),
  MTP speculative decoding with 3 drafts as Chrysalis uses, one slot, a 131K context, and prompt batches of 2,048 tokens
  (4,096 was 3% faster on prompts but doesn't fit beside that context).
- The same benchmark and PC as above. Chrysalis ran it twice and llama.cpp once; medians over every round, and over 10
  and 5 trials at 100K. Sanity checks: Chrysalis 24/24, llama.cpp 12/12.
- Strata's requests ask for a thinking budget of 0, which Strata reads as no cap and llama.cpp as no thinking at all, so
  the benchmark leaves it out and every engine thinks as Strata's does.
- **Share of this GPU's limit**: as above, with each engine's tokens per pass from its own log.

| | Chrysalis, 4.65-bit file | llama.cpp, the same file |
| --- | ---: | ---: |
| First token, short prompt | **0.12 s** | 0.31 s |
| First token, 4K prompt | **2.8 s** | 4.0 s |
| First token, 16K prompt | **11.6 s** | 16.8 s |
| First token, 100K prompt | **93.5 s** | 145.3 s |
| First token, follow-up in the 100K chat | **1.7 s** | 2.9 s |
| Writing, short prompt | **103.7 tok/s** | 68.4 tok/s |
| Writing, 4K prompt | **90.6 tok/s** | 57.3 tok/s |
| Writing, 16K prompt | **80.9 tok/s** | 51.5 tok/s |
| Writing, 100K prompt | **73.1 tok/s** | 34.7 tok/s |
| Writing, follow-up in the 100K chat | **74.3 tok/s** | 38.5 tok/s |
| Load time, to a first answer | **1.6 s** | 3.3 s |
| Peak private RAM | **0.5 GB** | 11.9 GB |
| Share of this GPU's limit, reading 100K | **62%** | 40% |
| Share of this GPU's limit, writing at 100K | **55%** | 27% |

</details>

## Chrysalis vs MLX and llama.cpp on a Mac

Qwen3.8-27B on a Mac Studio with an M4 Max: Chrysalis on its Metal engine, MLX and llama.cpp, each at its best on this
Mac.

<p align="center"><img src="media/mac-27b.png" width="860" alt="Qwen3.8-27B on an M4 Max in Chrysalis, MLX and llama.cpp: time to first token and writing speed at five prompt lengths, load time, memory, and each one's share of this GPU's limit"></p>

Chrysalis writes 1.5x faster than MLX up to 16K and 1.1x at 100K, 1.1-1.3x faster than llama.cpp, and answers a
short prompt 1.8x sooner than MLX and 2.5x sooner than llama.cpp. Up to 16K the three read prompts at about the same
speed; at 100K MLX reads 1.2x faster than the other two, then runs out of GPU memory on the follow-up in that chat.

<details><summary>How it was measured, and every number</summary>

Mac Studio, M4 Max with a 32-core GPU (410 GB/s, 25.9 TFLOPS FP16), 36 GB, macOS 27.2, one engine at a time.

- **Chrysalis**: the same 4.65-bit file as on the RX 7900, on its Metal engine, with MTP drafting 3 tokens.
- **MLX** 0.32.4 with mlx-lm 0.32.2 and mlx-community's 4-bit Qwen3.8-27B (4.5 bits per weight with its scales),
  through `mlx_lm.server`, which has no MTP for this model. Its 100K prompt runs out of GPU memory with the default
  prefill steps of 2,048, so that case ran with steps of 512; its follow-up re-read the whole chat and ran out of memory
  even then.
- **llama.cpp** build 11669 (Homebrew 0.7.0) with the same file as Chrysalis: every layer on the GPU, flash attention,
  MTP with 3 drafts as Chrysalis uses, one slot and a 131K context.
- The same benchmark as above, its thinking budget of 0 left out here too. Chrysalis ran it twice, MLX and llama.cpp
  once; medians over every round. Sanity checks: Chrysalis 24/24, MLX 12/12, llama.cpp 12/12.
- Memory is each server's peak footprint, macOS's count of the memory a process holds, its GPU buffers included.
  Chrysalis's holds snapshots of the conversation, which a follow-up reads back instead of the prompt.
- **Share of this GPU's limit**: as above, with Apple's published bandwidth for this chip and its GPU's FP16 rate, which
  counts the packed half-precision math an M4 runs at twice its FP32 rate.

| | Chrysalis, 4.65-bit file | MLX, 4-bit | llama.cpp, the same file |
| --- | ---: | ---: | ---: |
| First token, short prompt | **0.31 s** | 0.56 s | 0.78 s |
| First token, 4K prompt | 19.8 s | **19.5 s** | 20.7 s |
| First token, 16K prompt | **82.0 s** | 82.4 s | 88.8 s |
| First token, 100K prompt | 800.2 s | **666.1 s** | 818.9 s |
| First token, follow-up in the 100K chat | **14.2 s** | out of memory | 14.9 s |
| Writing, short prompt | **32.6 tok/s** | 21.3 tok/s | 29.4 tok/s |
| Writing, 4K prompt | **30.4 tok/s** | 20.4 tok/s | 27.0 tok/s |
| Writing, 16K prompt | **28.6 tok/s** | 19.5 tok/s | 25.2 tok/s |
| Writing, 100K prompt | **16.9 tok/s** | 15.2 tok/s | 14.8 tok/s |
| Writing, follow-up in the 100K chat | **21.6 tok/s** | out of memory | 17.2 tok/s |
| Load time, to a first answer | 7.5 s | **6.0 s** | 8.7 s |
| Peak memory footprint | 25.2 GB | 30.6 GB | **20.9 GB** |
| Share of this GPU's limit, reading 100K | 33% | **39%** | 32% |
| Share of this GPU's limit, writing at 100K | 35% | **68%** | 30% |

</details>

Flash-Next runs on this Mac too, keeping the experts its replies use most in memory and reading the others from the
SSD. In the same benchmark it answers a short prompt in 2.4 s and reads a 100K one in 255 s, and writes 6-8 tok/s.

## Editing artifacts by pointing at them

<p align="center"><img src="media/paint.webp" width="900" alt="An SVG cabin drawn live; pointing at the roof and asking to make it red changes only the roof; the paint bucket fills the door by hand"></p>

<sub>Artifacts render while they are written. Hovering over a part highlights it and clicking attaches it to your next
message, so "make this bright red" changes only that part. The door was filled with the paint bucket, which edits the
artifact directly without the model.</sub>

## SillyTavern character cards

<p align="center"><img src="media/characters.webp" width="900" alt="Seraphina's SillyTavern card dropped in: her greeting, a reply in her voice, her card"></p>

<sub>A SillyTavern card imported by drag and drop: the character's greeting, and speech and narration formatted the way
SillyTavern formats them. In group chats each speaker has their own portrait. Seraphina is SillyTavern's sample character
(AGPL-3.0), so this one recording is under their license.</sub>

## Phones and other devices

<p align="center"><img src="media/phone-pairing.png" width="760" alt="A phone connecting: the connect page, the code it shows, then the chat"></p>
<p align="center"><img src="media/connect-desktop.png" width="520" alt="Connect a device on the computer: the QR code and the phone's code with Allow"></p>

<sub>Scan the QR code, trust this computer's certificate authority once (the page shows how for each kind of device), and
allow the code the phone shows. The chat then runs over HTTPS. The authority is only valid for addresses on your
network, and only devices you allow get past the connect page.</sub>

## Also

- **Voice mode** that hears you talk over the reply
- **Tools**, MCP servers and sub-agents
- **Claude Code** and **OpenCode**, through OpenAI and Anthropic APIs
- The same chat on **llama.cpp, Ollama, LM Studio, KoboldCpp or vLLM**
- **Qwen3.8-27B** with its full 262K context: 104 tok/s on short prompts, 73 at 100K
- **Apple Silicon**: both models on a Mac with 32 GB or more, on Metal kernels written for its GPU; on an M4 Max the 27B
  writes 1.5x faster than MLX

## Run it

With an RX 7900 XTX, XT or GRE. Linux:

```sh
curl -fsSL https://github.com/michaelDinelle/chrysalis-inference/releases/latest/download/chrysalis-linux-x86_64.tar.gz | tar -xz
chrysalis/chrysalis.sh
```

Nothing is installed outside that folder. With 64 GB of RAM, each start asks which model to run; otherwise it runs the
27B. The first start downloads the model (16 GB for the 27B, 117 GB for Flash-Next) and prints the chat's address.
Chrysalis runs in that terminal, and Ctrl+C stops it.

Your laptop and phone use it too: `chrysalis.sh --host 0.0.0.0` puts it on your network. A laptop runs the same
Chrysalis, finds yours by itself, and asks in both terminals with the same code: answer y on each, once, and its apps
use your GPU at `http://127.0.0.1:8080`, as if it were there. A phone scans the QR code, trusts Chrysalis once, and gets the chat. No RX 7900 at all? You get the chat
alone, for Ollama, LM Studio or llama.cpp.

A Mac with Apple Silicon and 32 GB or more runs the models itself from a clone of this repository: the first
`./chrysalis.sh` gets Python and Bun into the folder and builds the Metal engines with Xcode's Metal compiler. The
release for a Mac, the same two lines with `chrysalis-macos-arm64.tar.gz`, is a laptop's chat for now; after that,
`chrysalis/Chrysalis.app` opens it in Terminal.

Windows: double-click [chrysalis.wsl](https://github.com/michaelDinelle/chrysalis-inference/releases/latest/download/chrysalis.wsl) (with WSL on: `wsl --install` as an administrator, once). Then Chrysalis is in the Start menu: its window is Chrysalis, asks there, and closing it stops it. On your network instead: `wsl -d Chrysalis --cd ~ -- chrysalis/chrysalis.sh --host 0.0.0.0` in PowerShell.

Docker, for the API alone (`http://localhost:8080/v1`):

```sh
docker run -d --device /dev/kfd --device /dev/dri -p 127.0.0.1:8080:8080 -v chrysalis:/models ghcr.io/michaeldinelle/chrysalis
```

## How it works

Chrysalis has its own engine for the RX 7900: hand-written kernels for each quantized format, MTP speculative decoding,
and a prompt cache that turns each agent turn into an append. Flash-Next's 24,576 experts are placed by how often its
replies use them: the most used in VRAM, following the conversation, and the rest in RAM, computed by the CPU. VRAM the
KV cache isn't using holds about 1,300 more experts, until a long context needs it back. Each machine's speeds are
measured once and kept, and experts move at fixed steps, so a run repeats to the bit: that is how every engine change
is checked against the last, and none may make any measure worse on any machine that runs it.

On a Mac the same files run on Metal kernels written for Apple's GPU, with the same sampler and MTP drafts. Flash-Next
keeps the experts its replies use most in memory and reads the others from the SSD as they are needed.

## Links

[X @michaeldinelle](https://x.com/michaeldinelle) · [Ko-fi](https://ko-fi.com/michaeldinelle) ·
[GitHub](https://github.com/michaelDinelle)

MIT licensed.

## Rewards

Every number here is held by the gate: a change is promoted only when at least one of them gets better and none gets worse, on every machine with a record. ↑ higher is better, ↓ lower is better.

### Speed

**Flash-Next on 7900 XTX**

|  |  | short | 4K | 16K deep | 100K deep | follow-up |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| First token | ↓ | 0.277 s | 1.98 s | 0.998 s | 1.18 s | 0.188 s |
| Reading the prompt | ↑ | 148 tok/s | 2,111 tok/s | 1,120 tok/s | 995 tok/s | 170 tok/s |
| Writing | ↑ | 78.7 tok/s | 79.7 tok/s | 90.0 tok/s | 83.5 tok/s | 89.9 tok/s |
| Longest gap between tokens, p99 | ↓ | 39.4 ms | 36.2 ms | 33.2 ms | 33.5 ms | 30.9 ms |
| Tokens per pass | ↑ | 2.17 | 2.06 | 2.46 | 2.46 | 2.41 |

**Flash-Next on M4 Max**

|  |  | short | 4K | 16K deep | 100K deep | follow-up |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| First token | ↓ | 3.13 s | 12.4 s | 12.0 s | 12.8 s | 1.66 s |
| Reading the prompt | ↑ | 12.2 tok/s | 330 tok/s | 90.0 tok/s | 84.6 tok/s | 17.7 tok/s |
| Writing | ↑ | 7.54 tok/s | 8.55 tok/s | 7.96 tok/s | 8.44 tok/s | 9.74 tok/s |
| Longest gap between tokens, p99 | ↓ | 482 ms | 513 ms | 633 ms | 336 ms | 511 ms |
| Tokens per pass | ↑ | 2.41 | 2.56 | 2.46 | 2.10 | 2.36 |

**27B on 7900 XTX**

|  |  | short | 4K | 16K deep | 100K deep | follow-up |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| First token | ↓ | 0.114 s | 2.80 s | 1.04 s | 1.95 s | 0.197 s |
| Reading the prompt | ↑ | 404 tok/s | 1,461 tok/s | 1,075 tok/s | 582 tok/s | 168 tok/s |
| Writing | ↑ | 103 tok/s | 82.1 tok/s | 107 tok/s | 89.5 tok/s | 109 tok/s |
| Longest gap between tokens, p99 | ↓ | 26.6 ms | 27.2 ms | 35.0 ms | 33.7 ms | 33.7 ms |
| Tokens per pass | ↑ | 2.74 | 2.22 | 3.05 | 3 | 3.67 |

**27B on M4 Max**

|  |  | short | 4K | 16K deep | 100K deep | follow-up |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| First token | ↓ | 0.344 s | 19.6 s | 6.20 s | 13.8 s | 0.573 s |
| Reading the prompt | ↑ | 102 tok/s | 207 tok/s | 173 tok/s | 78.0 tok/s | 47.4 tok/s |
| Writing | ↑ | 31.4 tok/s | 43.4 tok/s | 28.4 tok/s | 22.7 tok/s | 23.0 tok/s |
| Longest gap between tokens, p99 | ↓ | 73.4 ms | 75.2 ms | 87.4 ms | 131 ms | 260 ms |
| Tokens per pass | ↑ | 2.21 | 3.10 | 2.29 | 2.83 | 2.81 |

|  |  | Flash-Next · 7900 XTX | Flash-Next · M4 Max | 27B · 7900 XTX | 27B · M4 Max |
| --- | :---: | ---: | ---: | ---: | ---: |
| Load to a first answer | ↓ | 14.4 s | 6.20 s | 4.07 s | 5.88 s |

### Quality

|  |  | Flash-Next · 7900 XTX | Flash-Next · M4 Max | 27B · 7900 XTX | 27B · M4 Max |
| --- | :---: | ---: | ---: | ---: | ---: |
| Sanity tasks solved, of 12 | ↑ | 12 | 12 | 12 | 12 |
| KL vs Q8_0, mean | ↓ | 0.0339 | 0.0351 | 0.00335 | 0.00308 |
| KL vs Q8_0, p99.9 | ↓ | 0.594 | 0.849 | 0.0422 | 0.0357 |
| KL vs Q8_0, max | ↓ | 0.772 | 0.991 | 0.0439 | 0.0363 |
| Same top token as Q8_0 | ↑ | 93.4% | 93.5% | 97.0% | 97.3% |

### Memory and context

|  |  | Flash-Next · 7900 XTX | Flash-Next · M4 Max | 27B · 7900 XTX | 27B · M4 Max |
| --- | :---: | ---: | ---: | ---: | ---: |
| Context | ↑ | 257,920 tokens | 163,840 tokens | 278,144 tokens | 263,168 tokens |
| Prompt the follow-up reuses | ↑ | 99,930 tokens | 99,930 tokens | 99,930 tokens | 99,930 tokens |
| Peak RAM | ↓ | 47.4 GB | 28.6 GB | 3.96 GB | 25.1 GB |
| Lowest free system memory during a run | ↑ | 13.9 GB | 6.38 GB | 57.2 GB | 9.28 GB |
| Pool steps limited by memory | ↓ | 0 | 0 | — | — |

### Energy

|  |  | Flash-Next · 7900 XTX | Flash-Next · M4 Max | 27B · 7900 XTX | 27B · M4 Max |
| --- | :---: | ---: | ---: | ---: | ---: |
| GPU energy for one speed run | ↓ | 5.25 kJ | — | 5.30 kJ | — |

### Build

|  |  | Flash-Next · 7900 XTX | Flash-Next · M4 Max | 27B · 7900 XTX | 27B · M4 Max | 7900 XTX | M4 Max |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean build of both engines | ↓ | — | — | — | — | 8.03 s | 2.00 s |
| Engine library | ↓ | 1,506,144 bytes | 331,512 bytes | 1,098,368 bytes | 219,736 bytes | — | — |

### Velocity

|  |  | Flash-Next · 7900 XTX | Flash-Next · M4 Max | 27B · 7900 XTX | 27B · M4 Max |
| --- | :---: | ---: | ---: | ---: | ---: |
| Time for one speed run | ↓ | 31.8 s | 86.2 s | 21.2 s | 57.0 s |

### Against competitors

**Flash-Next on 7900 XTX**

|  |  | short | 4K |
| --- | :---: | ---: | ---: |
| First token, times sooner than Strata IQ3_S | ↑ | 2.31× | 1.72× |
| Writing, times faster than Strata IQ3_S | ↑ | 1.08× | 1.20× |
| First token, times sooner than Strata Q2_0 | ↑ | 1.50× | 1.53× |
| Writing, times faster than Strata Q2_0 | ↑ | 0.733× | 0.834× |

**Flash-Next on M4 Max**

|  |  | short | 4K |
| --- | :---: | ---: | ---: |
| First token, times sooner than Strata IQ3_S | ↑ | — | — |
| Writing, times faster than Strata IQ3_S | ↑ | — | — |
| First token, times sooner than Strata Q2_0 | ↑ | — | — |
| Writing, times faster than Strata Q2_0 | ↑ | — | — |

**27B on 7900 XTX**

|  |  | short | 4K |
| --- | :---: | ---: | ---: |
| First token, times sooner than llama.cpp | ↑ | 2.76× | 1.44× |
| Writing, times faster than llama.cpp | ↑ | 1.51× | 1.43× |

**27B on M4 Max**

|  |  | short | 4K |
| --- | :---: | ---: | ---: |
| First token, times sooner than llama.cpp | ↑ | — | — |
| Writing, times faster than llama.cpp | ↑ | — | — |

|  |  | Flash-Next · 7900 XTX | Flash-Next · M4 Max |
| --- | :---: | ---: | ---: |
| KL vs Q8_0, times lower than Strata IQ3_S | ↑ | 1.62× | — |
| KL vs Q8_0, times lower than Strata Q2_0 | ↑ | 5.51× | — |
