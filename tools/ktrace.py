"""Per-kernel time inside the last `window` seconds of a rocprofv3 kernel trace (e.g. a timed decode loop).
usage: data/rocm/bin/rocprofv3 --kernel-trace --output-format csv -d build/prof -o run -- <command>
       tools/ktrace.py build/prof <window seconds> [steps in the window] [rows]"""
import csv, glob, sys, collections
f = glob.glob(f"{sys.argv[1]}/**/*kernel_trace.csv", recursive=True)[0]
window = float(sys.argv[2]) * 1e9
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 1
rows = list(csv.DictReader(open(f)))
end = max(int(r["End_Timestamp"]) for r in rows)
rows = [r for r in rows if int(r["Start_Timestamp"]) >= end - window]
agg = collections.defaultdict(lambda: [0, 0])
for r in rows:
    a = agg[r["Kernel_Name"]]; a[0] += int(r["End_Timestamp"]) - int(r["Start_Timestamp"]); a[1] += 1
busy = sum(v[0] for v in agg.values())
span = max(int(r["End_Timestamp"]) for r in rows) - min(int(r["Start_Timestamp"]) for r in rows)
print(f"per step: wall {span/steps/1e3:.0f} us, kernels busy {busy/steps/1e3:.0f} us, idle/gaps {(span-busy)/steps/1e3:.0f} us, launches {len(rows)/steps:.0f}")
for name, (ns, n) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:int(sys.argv[4]) if len(sys.argv) > 4 else 16]:
    print(f"  {ns/steps/1e3:8.1f} us/step  {n/steps:6.1f} calls  avg {ns/n/1e3:7.1f} us  {name[:80]}")
