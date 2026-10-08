"""OpenCode-style compaction against a running server: after a short agent session, send the compaction
request (the whole conversation serialized into one message, no tools) and check it is served from cache."""
import json, os, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent_sim as sim   # runs one agent session on import and leaves it in sim.messages

URL = sim.URL


def serialize(m):
    if m["role"] == "user":
        return "[User]: " + m["content"]
    if m["role"] == "tool":
        return "[Tool result]: " + m["content"][:2000]
    out = []
    if m.get("reasoning_content"):
        out.append("[Assistant reasoning]: " + m["reasoning_content"])
    if m.get("content"):
        out.append("[Assistant]: " + m["content"])
    for c in m.get("tool_calls") or []:
        out.append(f"[Assistant tool call]: {c['function']['name']}({c['function']['arguments']})")
    return "\n".join(out)


conversation = "\n\n".join(serialize(m) for m in sim.messages[1:])
system = ("You are a context summarization agent. You are given a conversation between a user and an agent. Your goal is "
          "to produce a structured summary matching the format specified so another coding agent can continue the work.")
prompt = (f"Here is the conversation so far:\n\n<conversation>\n{conversation}\n</conversation>\n\nCreate a new anchored summary "
          "from the conversation history in the <conversation> tags above so another coding agent can continue the work.\n\n"
          "## Goal\n## Progress\n## Key facts")
body = json.dumps({"model": "qwen3.8-27b", "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                   "max_tokens": 1500}).encode()
t0 = time.time()
r = json.load(urllib.request.urlopen(urllib.request.Request(URL + "/chat/completions", body, {"Content-Type": "application/json"}), timeout=3600))
u = r["usage"]
print(f"\ncompaction: prompt {u['prompt_tokens']} cached {u['prompt_tokens_details']['cached_tokens']} "
      f"total {time.time() - t0:.1f}s\nsummary:\n{r['choices'][0]['message']['content'][:1200]}")
ok = u["prompt_tokens_details"]["cached_tokens"] > 0.8 * u["prompt_tokens"]
print("served from cache" if ok else "NOT served from cache")
sys.exit(0 if ok else 1)
