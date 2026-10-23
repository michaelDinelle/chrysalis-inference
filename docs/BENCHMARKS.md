# Chrysalis vs Strata

One PC: Radeon RX 7900 XTX (24 GB), Ryzen 7 7800X3D, 64 GB DDR5, Linux. One model: Qwen3.8-Flash-Next. Measured
2026-10-22.

- **Strata** 0.2.2 (engine 0.1.47, its `linux-x86_64-hip-rx7900` build), installed by its own `install.sh` and
  `strata launch`, 128K context, its defaults otherwise. Two of its sizes: **IQ3_S**, the best its setup offers a 64 GB
  PC ("matches the full model on the published tests"), and **Q2_0**, its fastest.
- **Chrysalis** as it runs every day (`chrysalis start`), its one Flash-Next file.

## Speed

|  | Chrysalis | Strata IQ3_S | Strata Q2_0 |
| --- | ---: | ---: | ---: |
| First token, 4K-token prompt | **1.8 s** | 5.8 s | 5.5 s |
| First token, 16K-token prompt | **5.9 s** | 20.9 s | 20.0 s |
| First token, 100K-token prompt | **38 s** | 131 s | 130 s |
| First token, a follow-up in that 100K chat | **1.0 s** | 2.8 s | 2.6 s |
| Writing, short prompt | 66 tok/s | 65 tok/s | **91 tok/s** |
| Writing, after a 16K prompt | 46 tok/s | 63 tok/s | **96 tok/s** |
| Writing, after a 100K prompt | 49 tok/s | 61 tok/s | **64 tok/s** |
| Strata's 12 sanity checks | 12 / 12 | 12 / 12 | 12 / 12 |

Medians. The 4K, 16K and short rows and the sanity checks are **Strata's standard benchmark** (`strata bench run`,
protocol `strata-qwen38-thinking-v1-draft.1`): its frozen prompts, Qwen's thinking defaults (temperature 1, top-p 0.95,
top-k 20, `reasoning_effort: xhigh`), 512-token caps, three repetitions. Strata ran it itself. Its runner refuses a
server that is not Strata, so Chrysalis ran the same requests through Strata's own code ([`replay.py`](benchmarks/replay.py):
its fixture loader and manifest check, request body, streaming client and scoring; only the server check left out).

Every engine is timed by the client the same way: the first token is the first streamed event, writing speed is the
tokens after it over the time to the last. Strata's own engine timings agree with that clock to within 1%
(`strata-*.json` against `strata-*-client.json`).

The 100K rows are [`longctx.py`](benchmarks/longctx.py): 98,878 tokens of records built as Strata's long prompts are,
then a follow-up of about 1,100 new tokens in the same conversation, two trials each.

## Distance from the full model

|  | Chrysalis | Strata IQ3_S | Strata Q2_0 |
| --- | ---: | ---: | ---: |
| Mean KL from the Q8_0 model, 1,082 tokens | **0.032** | 0.055 | 0.187 |
| Same top token as the Q8_0 model | **92.8%** | 90.9% | 84.1% |
| Mean KL, 166 tokens of code explanation | **0.015** | 0.029 | 0.076 |

Each model's next-token distribution at every position of the same text, against llama.cpp running the Q8_0 original.
Strata's files ran in llama.cpp (on the CPU, Strata's weights exactly). Chrysalis ran in Chrysalis, one token at a time
as it writes a reply ([`tools_fn/compare_fn.py`](../tools_fn/compare_fn.py)), so its number also carries its own
kernels' rounding. The text is [`kl-text.txt`](benchmarks/kl-text.txt): prose, a technical explanation and Python.

## Files

[`benchmarks/`](benchmarks): Strata's two reports as its runner wrote them (`strata-q2_0.json`, `strata-iq3_s.json`),
the same requests timed by the client clock (`*-client.json`), Chrysalis's run (`chrysalis-standard.json`), the 100K
runs (`*-longctx.json`), the KL numbers (`kl.json`) and the chart's script ([`chart.py`](benchmarks/chart.py)).
