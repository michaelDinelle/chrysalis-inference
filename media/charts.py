import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUMBERS = f"{ROOT}/media/charts.json"
README = f"{ROOT}/README.md"
W, LABEL_W, ROW, BAR, GROUP_GAP = 980, 262, 21, 12, 9
INK = ("#1d181a", "#6e6164", "#e5d6da")
FILL = {"Chrysalis": "#e24d85", "Strata IQ3_S": "#4d4d4d", "Strata Q2_0": "#bdbdbd", "llama.cpp": "#4d4d4d", "MLX": "#bdbdbd"}
CASES = ["Short", "4K", "16K", "100K", "Follow-up"]
PROMPTS = ["short prompt", "4K prompt", "16K prompt", "100K prompt", "follow-up in the 100K chat"]
MACHINES = {"7900 XTX": "RX 7900 XTX", "RTX 5070": "RTX 5070", "M4 Max": "M4 Max"}
MODELS = {"fn": ("flash-next", "Qwen3.8-Flash-Next", [("Chrysalis", "3.8-bit"), ("Strata IQ3_S", "3.3-bit"), ("Strata Q2_0", "2.3-bit")]),
          "27b": ("27b", "Qwen3.8-27B", [("Chrysalis", "4.65-bit file"), ("llama.cpp", "the same file"), ("MLX", "4-bit")])}


def fmt_s(v):
    return f"{v:.2f} s" if v < 1 else f"{v:.1f} s" if v < 100 else f"{v:.0f} s"


def fmt_kl(v):
    return f"{v:.4f}" if v < 0.1 else f"{v:.3f}"


def fmt_gb(v):
    return f"{v:.1f} GB"


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;")


def block(out, y, title, sub, heads, data, cols, scale=None):
    out.append(f'<text x="0" y="{y}" font-size="15" font-weight="600" fill="{INK[0]}">{esc(title)}</text>')
    out.append(f'<text x="0" y="{y + 18}" font-size="12" fill="{INK[1]}">{esc(sub)}</text>')
    y += 44
    cw = (W - LABEL_W) / len(cols)
    for j, h in enumerate(heads):
        out.append(f'<text x="{LABEL_W + j * cw:.1f}" y="{y}" font-size="12.5" font-weight="600" fill="{INK[0]}">{esc(h)}</text>')
    y += 10
    groups = []
    for d in data:
        if not groups or groups[-1][-1]["machine"] != d["machine"]:
            groups.append([])
        groups[-1].append(d)
    for gi, grp in enumerate(groups):
        if gi:
            out.append(f'<line x1="0" x2="{W}" y1="{y + GROUP_GAP / 2:.1f}" y2="{y + GROUP_GAP / 2:.1f}" stroke="{INK[2]}"/>')
            y += GROUP_GAP
        for d in grp:
            fill = FILL[d["engine"]]
            out.append(f'<rect x="0" y="{y + (ROW - 11) / 2:.1f}" width="11" height="11" rx="3" fill="{fill}"/>')
            out.append(f'<text x="18" y="{y + ROW / 2 + 4.5:.1f}" font-size="13" fill="{INK[0]}"><tspan font-weight="600">{esc(d["engine"])}</tspan>'
                       f'<tspan dx="5">{esc(d["machine"])} ({esc(d["backend"])})</tspan></text>')
            for j, (key, idx, fmtv, lower) in enumerate(cols):
                v = d[key][idx] if idx is not None else d[key]
                x0 = LABEL_W + j * cw
                if not isinstance(v, (int, float)):
                    out.append(f'<text x="{x0:.1f}" y="{y + ROW / 2 + 4:.1f}" font-size="11.5" fill="{INK[1]}">{esc(v or "not reported")}</text>')
                    continue
                known = [e[key][idx] if idx is not None else e[key] for e in data]
                known = [k for k in known if isinstance(k, (int, float))]
                mates = [e[key][idx] if idx is not None else e[key] for e in grp]
                mates = [k for k in mates if isinstance(k, (int, float))]
                best = len(mates) > 1 and v == (min(mates) if lower else max(mates))
                w = max(3.0, (cw - 66) * v / (scale or max(known)))
                by = y + (ROW - BAR) / 2
                out.append(f'<path d="M{x0:.1f},{by:.1f} h{max(0.0, w - 4):.1f} a4,4 0 0 1 4,4 v{BAR - 8} a4,4 0 0 1 -4,4 h-{max(0.0, w - 4):.1f} z" fill="{fill}"/>')
                out.append(f'<text x="{x0 + w + 5:.1f}" y="{by + BAR - 2:.1f}" font-size="11.5" fill="{INK[0] if d["engine"] == "Chrysalis" else INK[1]}" '
                           f'font-weight="{700 if best else 400}">{esc(fmtv(v))}</text>')
            y += ROW
    return y + 14


def svg(title, legend, data):
    out = [f'<text x="0" y="16" font-size="19" font-weight="700" fill="{INK[0]}">{esc(title)}</text>']
    lx = 0
    for name, bits in legend:
        out.append(f'<rect x="{lx}" y="32" width="12" height="12" rx="3" fill="{FILL[name]}"/>')
        out.append(f'<text x="{lx + 18}" y="43" font-size="13" fill="{INK[0]}"><tspan font-weight="600">{esc(name)}</tspan>'
                   f'<tspan dx="5" fill="{INK[1]}">· {esc(bits)}</tspan></text>')
        lx += 18 + int(7.4 * len(name) + 6.4 * (len(bits) + 3)) + 34
    y = block(out, 84, "Time to first token", "lower is better · seconds", CASES, data, [("ttft", i, fmt_s, True) for i in range(5)])
    out.append(f'<line x1="0" x2="{W}" y1="{y}" y2="{y}" stroke="{INK[2]}"/>')
    y = block(out, y + 30, "Writing speed", "higher is better · tokens per second", CASES, data,
              [("tps", i, lambda v: f"{v:.1f}", False) for i in range(5)])
    out.append(f'<line x1="0" x2="{W}" y1="{y}" y2="{y}" stroke="{INK[2]}"/>')
    y = block(out, y + 30, "Quality, load and memory", "lower is better", ["Error vs Q8_0 (KL)", "Load to a first answer", "Peak memory"], data,
              [("kl", None, fmt_kl, True), ("load", None, fmt_s, True), ("ram", None, fmt_gb, True)])
    h = y - 4
    pad = 30
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W + 2 * pad:.0f}" height="{h + 2 * pad:.0f}" '
            f'font-family="-apple-system,Segoe UI,Helvetica,Arial,sans-serif">\n'
            f'<rect x="0.5" y="0.5" width="{W + 2 * pad - 1:.0f}" height="{h + 2 * pad - 1:.0f}" rx="14" fill="#ffffff" stroke="#e5d6da"/>\n'
            f'<g transform="translate({pad},{pad})">\n' + "\n".join(out) + "\n</g>\n</svg>\n")


def png(name, card, digest):
    from PIL import Image, PngImagePlugin
    with tempfile.TemporaryDirectory() as tmp:
        with open(f"{tmp}/card.svg", "w") as f:
            f.write(card)
        subprocess.run(["rsvg-convert", "-z", "2", f"{tmp}/card.svg", "-o", f"{tmp}/card.png"], check=True)
        image = Image.open(f"{tmp}/card.png").convert("RGBA").quantize(256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE)
    info = PngImagePlugin.PngInfo()
    info.add_text("numbers", digest)
    image.save(f"{ROOT}/media/{name}.png", optimize=True, pnginfo=info)


def stamp(path):
    with open(path, "rb") as f:
        data = f.read()
    at = 8
    while at + 8 <= len(data):
        size, kind = struct.unpack(">I4s", data[at:at + 8])
        if kind == b"tEXt":
            key, _, value = data[at + 8:at + 8 + size].partition(b"\0")
            if key == b"numbers":
                return value.decode()
        at += size + 12
    return None


def table(data):
    rows = [(f"First token, {c}", "ttft", i, fmt_s) for i, c in enumerate(PROMPTS)]
    rows += [(f"Writing, {c}", "tps", i, lambda v: f"{v:.1f} tok/s") for i, c in enumerate(PROMPTS)]
    rows += [("Error vs Q8_0, mean KL over 1,082 tokens", "kl", None, fmt_kl), ("Load time, to a first answer", "load", None, fmt_s),
             ("Peak memory", "ram", None, fmt_gb)]
    lines = ["| | " + " | ".join(f'{d["engine"]} {d["machine"]} ({d["backend"]})' for d in data) + " |", "| --- |" + " ---: |" * len(data)]
    for label, key, i, fmtv in rows:
        vals = [d[key][i] if i is not None else d[key] for d in data]
        lines.append(f"| {label} | " + " | ".join(fmtv(v) if isinstance(v, (int, float)) else v or "not reported" for v in vals) + " |")
    return "\n".join(lines) + "\n"


def tables(numbers, text):
    written = 0
    for key, (_, title, _) in MODELS.items():
        for machine in dict.fromkeys(d["machine"] for d in numbers[key]):
            heading = f"**{title}, {MACHINES[machine]}**\n\n"
            body = table([d for d in numbers[key] if d["machine"] == machine])
            text, found = re.subn(re.escape(heading) + r"(?:\|.*\n)*", lambda _: heading + body, text)
            if found != 1:
                sys.exit(f"README.md needs the heading {heading.strip()} once, and has it {found} times")
            written += body.count("\n")
    return text, written


def read():
    with open(NUMBERS, "rb") as f:
        raw = f.read()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def check():
    numbers, digest = read()
    with open(README) as f:
        text = f.read()
    want, rows = tables(numbers, text)
    problems = []
    if want != text:
        have, need = text.splitlines() + [""], want.splitlines() + [""]
        line = next((a, b) for a, b in zip(have, need) if a != b)
        problems.append(f"README.md has {line[0]!r} where media/charts.json gives {line[1]!r}")
    if sum(line.startswith("|") for line in text.splitlines()) != rows:
        problems.append("README.md has a table row that media/charts.json doesn't generate")
    for name, *_ in MODELS.values():
        if stamp(f"{ROOT}/media/{name}.png") != digest:
            problems.append(f"media/{name}.png wasn't drawn from media/charts.json")
    if problems:
        sys.exit("\n".join(problems) + "\nrun: data/venv/bin/python media/charts.py")


def main():
    numbers, digest = read()
    for key, (name, title, legend) in MODELS.items():
        png(name, svg(title, legend, numbers[key]), digest)
    with open(README) as f:
        text = f.read()
    with open(README, "w") as f:
        f.write(tables(numbers, text)[0])


if __name__ == "__main__":
    check() if "--check" in sys.argv else main()
