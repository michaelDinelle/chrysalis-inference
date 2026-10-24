import threading

TTFT_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300)


def _labels(d):
    return "{" + ",".join(f'{k}="{str(v).replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}"'
                          for k, v in d.items()) + "}" if d else ""


def _num(v):
    return repr(float(v)) if isinstance(v, float) else str(int(v))


class Metrics:
    COUNTERS = {
        "chrysalis_replies_total": "Replies written, by model and finish reason (stop, length, tool_calls, client: stopped by the client)",
        "chrysalis_prompt_tokens_total": "Prompt tokens of the replies",
        "chrysalis_prompt_cached_tokens_total": "Prompt tokens served from the prompt cache instead of read",
        "chrysalis_completion_tokens_total": "Tokens written",
        "chrysalis_decode_steps_total": "Decode steps (a step writes 1 token plus the drafts it accepts)",
        "chrysalis_prefill_seconds_total": "Time from a request to its first decode step",
        "chrysalis_decode_seconds_total": "Time spent writing replies after their prefill",
        "chrysalis_model_loads_total": "Model loads and switches after the first, by the model loaded",
        "chrysalis_model_load_seconds_total": "Time those loads took",
        "chrysalis_http_errors_total": "Requests answered with an error status, by status",
    }

    def __init__(self):
        self.lock = threading.Lock()
        self.counts = {}
        self.ttft = {}

    def add(self, name, value=1, **labels):
        key = (name, tuple(sorted(labels.items())))
        with self.lock:
            self.counts[key] = self.counts.get(key, 0) + value

    def reply(self, model, info, steps, finish):
        self.add("chrysalis_replies_total", model=model, finish=finish)
        self.add("chrysalis_prompt_tokens_total", info["prompt_tokens"], model=model)
        self.add("chrysalis_prompt_cached_tokens_total", info["cached_tokens"], model=model)
        self.add("chrysalis_completion_tokens_total", info["completion_tokens"], model=model)
        self.add("chrysalis_decode_steps_total", steps, model=model)
        self.add("chrysalis_prefill_seconds_total", info["prefill_s"], model=model)
        self.add("chrysalis_decode_seconds_total", info["decode_s"], model=model)
        with self.lock:
            h = self.ttft.setdefault(model, [0] * (len(TTFT_BUCKETS) + 2))
            for i, b in enumerate(TTFT_BUCKETS):
                h[i] += info["ttft"] <= b
            h[-2] += 1
            h[-1] += info["ttft"]

    def render(self, gauges):
        out = []
        with self.lock:
            counts, ttft = dict(self.counts), {m: list(h) for m, h in self.ttft.items()}
        for name, help_ in self.COUNTERS.items():
            out += [f"# HELP {name} {help_}", f"# TYPE {name} counter"]
            out += [f"{name}{_labels(dict(k))} {_num(v)}" for (n, k), v in sorted(counts.items()) if n == name]
        name = "chrysalis_time_to_first_token_seconds"
        out += [f"# HELP {name} Time from a request to its first token", f"# TYPE {name} histogram"]
        for model, h in sorted(ttft.items()):
            for i, b in enumerate(TTFT_BUCKETS):
                out.append(f"{name}_bucket{_labels({'model': model, 'le': b})} {h[i]}")
            out += [f"{name}_bucket{_labels({'model': model, 'le': '+Inf'})} {h[-2]}",
                    f"{name}_count{_labels({'model': model})} {h[-2]}", f"{name}_sum{_labels({'model': model})} {_num(float(h[-1]))}"]
        for name, help_, samples in gauges:
            kind = "counter" if name.endswith("_total") else "gauge"
            out += [f"# HELP {name} {help_}", f"# TYPE {name} {kind}"]
            out += [f"{name}{_labels(lab)} {_num(v)}" for lab, v in samples if v is not None]
        return "\n".join(out) + "\n"
