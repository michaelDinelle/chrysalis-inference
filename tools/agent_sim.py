"""A miniature OpenCode: drives the Chrysalis server through a real agent loop (OpenCode's system prompt and tool
schemas, read-only tools executed for real in this repo) and reports cache reuse and latency per turn.
usage: agent_sim.py ["task"] [--url=http://127.0.0.1:8080/v1] [--model=qwen3.8-27b]"""
import glob as globmod
import json
import os
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--url=")), "http://127.0.0.1:8080/v1")
MODEL = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--model=")), "qwen3.8-27b")
TASK = next((a for a in sys.argv[1:] if not a.startswith("--")),
            "How does Chrysalis's decode attention kernel avoid LDS, and how many lines is engine/attn.hip? Be brief.")
TOOLS = json.load(open(os.path.join(ROOT, "tools/opencode_tools.json")))
SYSTEM = open(os.path.join(ROOT, "tools/opencode_default_prompt.txt")).read() + f"""
You are powered by the model named {MODEL}. The exact model ID is chrysalis/{MODEL}
Here is some useful information about the environment you are running in:
<env>
  Working directory: {ROOT}
  Workspace root folder: {ROOT}
  Is directory a git repo: no
  Platform: linux
  Today's date: {time.strftime('%a %b %d %Y')}
</env>"""


def run_tool(name, args):
    try:
        if name == "read":
            p = args["filePath"]
            if os.path.isdir(p):
                return "\n".join(sorted(e + ("/" if os.path.isdir(os.path.join(p, e)) else "") for e in os.listdir(p)))
            lines = open(p, errors="replace").read().split("\n")
            off = int(args.get("offset", 1) or 1)
            lim = int(args.get("limit", 2000) or 2000)
            return "\n".join(f"{i}: {l}" for i, l in enumerate(lines[off - 1:off - 1 + lim], off))
        if name == "glob":
            return "\n".join(sorted(globmod.glob(os.path.join(args.get("path") or ROOT, args["pattern"]), recursive=True))[:100]) or "No files found"
        if name == "grep":
            r = subprocess.run(["grep", "-rnE", args["pattern"], args.get("path") or ROOT] + (["--include", args["include"]] if args.get("include") else []),
                               capture_output=True, text=True, timeout=20)
            return r.stdout[:20000] or "No matches found"
        if name == "bash":
            cmd = args["command"]
            if any(w in cmd for w in ("rm ", ">", "mv ", "kill", "sudo", "curl", "wget")):
                return "Error: this simulation only allows read-only commands"
            r = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True, timeout=30)
            return (r.stdout + r.stderr)[:20000]
        if name == "todowrite":
            return json.dumps(args.get("todos", []))
        return f"Error: {name} is disabled in this read-only simulation"
    except Exception as e:
        return f"Error: {e}"


def chat(messages):
    body = json.dumps({"model": MODEL, "messages": messages, "tools": TOOLS, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(URL + "/chat/completions", body, {"Content-Type": "application/json"})
    t0, t_first = time.time(), None
    reasoning, content, calls, usage = [], [], [], {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            usage = d.get("usage") or usage
            for ch in d.get("choices", []):
                dl = ch.get("delta", {})
                if (dl.get("reasoning_content") or dl.get("content") or dl.get("tool_calls")) and t_first is None:
                    t_first = time.time()
                reasoning.append(dl.get("reasoning_content") or "")
                content.append(dl.get("content") or "")
                calls += dl.get("tool_calls") or []
    return "".join(reasoning), "".join(content), calls, usage, (t_first or time.time()) - t0, time.time() - t0


messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TASK}]
print(f"task: {TASK}\n")
for turn in range(1, 25):
    reasoning, content, calls, usage, ttft, total = chat(messages)
    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    gen = usage.get("completion_tokens", 0)
    print(f"turn {turn:2d}: prompt {usage.get('prompt_tokens', 0):6d}  cached {cached:6d}  ttft {ttft:5.2f}s  "
          f"gen {gen:5d} tokens in {total - ttft:5.1f}s ({gen / max(total - ttft, 1e-6):5.1f} tok/s)  "
          f"calls: {', '.join(c['function']['name'] for c in calls) or '-'}")
    msg = {"role": "assistant", "content": content, "reasoning_content": reasoning}
    if calls:
        msg["tool_calls"] = calls
    messages.append(msg)
    if not calls:
        print("\nanswer:\n" + content)
        break
    for c in calls:
        out = run_tool(c["function"]["name"], json.loads(c["function"]["arguments"] or "{}"))
        messages.append({"role": "tool", "tool_call_id": c["id"], "content": out})
