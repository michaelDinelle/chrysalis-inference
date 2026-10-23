"""Chrysalis vs Strata, one SVG per theme, from the measured results (same PC, same model, Strata's fixtures)."""
import json, os, statistics
R = os.path.dirname(os.path.abspath(__file__))
def std(name): return json.load(open(f"{R}/{name}"))["summary"]
def lc(name):
    rows = json.load(open(f"{R}/{name}"))
    return (statistics.median(r["long"]["ttft_s"] for r in rows), statistics.median(r["followup"]["ttft_s"] for r in rows))
kl = json.load(open(f"{R}/kl.json"))
eng = [("Chrysalis", std("chrysalis-standard.json"), lc("chrysalis-longctx.json"), kl["chrysalis"]["long"]["mean_kl"]),
       ("Strata IQ3_S", std("strata-iq3_s-client.json"), lc("strata-iq3_s-longctx.json"), kl["strata_iq3_s"]["long"]["mean_kl"]),
       ("Strata Q2_0", std("strata-q2_0-client.json"), lc("strata-q2_0-longctx.json"), kl["strata_q2_0"]["long"]["mean_kl"])]
rows = [  # title, unit, lower is better, value of each engine
    ("Reads a 16K-token prompt", "s", True, [e[1]["prompt-16k"]["ttft_s"] for e in eng]),
    ("Reads a 100K-token prompt", "s", True, [e[2][0] for e in eng]),
    ("Answers a follow-up at 100K", "s", True, [e[2][1] for e in eng]),
    ("Error vs the full model (KL)", "", True, [e[3] for e in eng]),
    ("Writes", "tok/s", False, [e[1]["prompt-16k"]["decode_tps"] for e in eng]),
]
def svg(dark):
    fg, mute, grid = ("#efe4e7", "#9c8e92", "#2d2628") if dark else ("#1d181a", "#6e6164", "#e5d6da")
    cols = ["#e24d85", "#8a8a8a", "#5d5d5d"] if dark else ["#e24d85", "#9a9a9a", "#c4c4c4"]
    W, x0, bw, bh, gap, rowh = 860, 250, 470, 16, 5, 98
    H = 64 + rowh * len(rows)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="-apple-system,Segoe UI,Helvetica,Arial,sans-serif">']
    # legend
    lx = x0
    for (name, *_), c in zip(eng, cols):
        o.append(f'<rect x="{lx}" y="14" width="12" height="12" rx="3" fill="{c}"/><text x="{lx + 18}" y="25" font-size="13" fill="{fg}">{name}</text>')
        lx += 150
    for k, (title, unit, low, vals) in enumerate(rows):
        y = 52 + k * rowh
        o.append(f'<text x="0" y="{y + 26}" font-size="15" font-weight="600" fill="{fg}">{title}</text>')
        o.append(f'<text x="0" y="{y + 46}" font-size="12" fill="{mute}">{"lower is better" if low else "higher is better"}</text>')
        top = max(vals)
        best = min(vals) if low else max(vals)
        for j, (v, c) in enumerate(zip(vals, cols)):
            by = y + 8 + j * (bh + gap)
            w = max(3, bw * v / top)
            label = f"{v:.3f}" if unit == "" else f"{v:.1f} {unit}" if v < 100 else f"{v:.0f} {unit}"
            o.append(f'<rect x="{x0}" y="{by}" width="{w:.1f}" height="{bh}" rx="4" fill="{c}"/>')
            o.append(f'<text x="{x0 + w + 8:.1f}" y="{by + 12.5}" font-size="12.5" fill="{fg if v == best else mute}" font-weight="{600 if v == best else 400}">{label}</text>')
        o.append(f'<line x1="0" x2="{W}" y1="{y + rowh - 10}" y2="{y + rowh - 10}" stroke="{grid}"/>' if k < len(rows) - 1 else "")
    o.append("</svg>")
    return "\n".join(o)
for dark, name in ((True, "dark"), (False, "light")):
    open(os.path.join(R, f"../media/vs-strata-{name}.svg"), "w").write(svg(dark))
